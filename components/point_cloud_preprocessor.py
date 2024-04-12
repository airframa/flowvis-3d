import os
import open3d as o3d
import numpy as np
import copy
import numba
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from components.point_cloud_visualizer import PointCloudVisualizer
from components.point_cloud_utils import calculate_point_cloud_metrics
        

class PointCloudPreprocessor():
    def __init__(self, input_cloud):
        """
        Initialize the PointCloudPreprocessor.

        Args:
            data (str or o3d.geometry.PointCloud): Either the path to the .ply file containing the point cloud data
                or the loaded point cloud itself.
            point_density_threshold (float, optional): The target point density for voxel downsampling. Default is 1.0.
        """
        # We retain copies of the input cloud to allow for multiple class instances in the main.ipynb without modifying the input cloud (experimentation phase)
        if isinstance(input_cloud, str):
            self.input_cloud = copy.deepcopy(o3d.io.read_point_cloud(input_cloud))          
        elif isinstance(input_cloud, o3d.geometry.PointCloud):
            self.input_cloud = copy.deepcopy(input_cloud)
        else:
            raise ValueError("Invalid source type. Provide a path to a PLY file or an existing point cloud.")
        
        # Initialize color masks
        self.blue_points_mask = None
        self.black_points_mask = None
        self.red_points_mask = None
        self.green_points_mask = None

        # Initialize discrepancy points lists
        self.combined_indices = None
        self.exact_match_indices = None

    def preprocess_data(self, point_density_threshold=2.0, voxel=True, normals=True, normals_check=False, save=False, output=True):
        """
        Performs preprocessing on the loaded point cloud data, which may include downsampling, normal estimation, and normal orientation. 
        This prepares the point cloud for further processing and analysis.

        Args:
            point_density_threshold (float): The desired point density to achieve after voxel downsampling.
            normals (bool): If True, calculate and orient normals for the point cloud.
            normals_check (bool): If True and normals are calculated, perform a consistency check on normals.
            save (bool): If True, save the preprocessed point cloud to disk.

        Returns:
            o3d.geometry.PointCloud: The preprocessed point cloud.
        """

        # Calculate metrics such as point density and bounding box dimensions for the input cloud.
        metrics = calculate_point_cloud_metrics(self.input_cloud)

        # Calculate the optimal voxel size based on the target point density threshold.
        voxel_size = self.calculate_voxel_size(metrics, point_density_threshold)
        # print(f'Voxel size: {voxel_size}')

        # Log the number of points in the point cloud before preprocessing.
        npoints_initial = len(self.input_cloud.points)

        # If a valid voxel size is calculated, downsample the point cloud to reduce its density.
        if voxel_size is not None and voxel==True:
             self.input_cloud = self.input_cloud.voxel_down_sample(voxel_size)

        # Log the number of points in the point cloud after downsampling.
        npoints_after = len(self.input_cloud.points)

        # If normals are to be computed, estimate and orient them for the point cloud.
        if normals:
            # Compute point normals using a hybrid approach considering both radius and nearest neighbors.
            radius_normal = 2.0
            self.input_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=10))

            # Orient normals to ensure consistency based on the surrounding tangent planes.
            self.input_cloud.orient_normals_consistent_tangent_plane(k=10)

            # Convert normals to a NumPy array for potential further manipulation.
            normals = np.asarray(self.input_cloud.normals)
            
            # If a normals consistency check is requested, perform the check and correct inconsistencies.
            if normals_check:
                self.check_inconsistencies()

            # Update the normals in the point cloud with any potential corrections made.
            self.input_cloud.normals = o3d.utility.Vector3dVector(normals)

        # If the save flag is set, write the preprocessed point cloud to a file for persistence.
        if save:
            # Replace with a path relevant to your environment or parameterize the output directory.
            o3d.io.write_point_cloud("D:/flowvis_data/pcd_preprocessed.ply", self.input_cloud)

        # Instantiate visualizer to display the segmentation result
        if output:
            print(f"Number of points before preprocessing: {npoints_initial}")
            print(f"Number of points after preprocessing: {npoints_after}\n")
            print("Preprocessed point cloud:")
            PointCloudVisualizer(self.input_cloud)

        # Return the preprocessed point cloud.
        return self.input_cloud


    def calculate_voxel_size(self, metrics, point_density_threshold):
        """
        Calculate the voxel size to reach the target point density.

        Args:
            point_cloud (o3d.geometry.PointCloud): The input 3D point cloud.

        Returns:
            float: The calculated voxel size, or None if no downsampling is needed.
        """

        # If the point cloud density is below the target density, no downsampling is needed
        if metrics["Point Density"]  < 1e-2:
            return None  # Return None to indicate no downsampling is needed

        # Calculate the volume of the point cloud's bounding box
        bounding_box_volume = np.prod(metrics["Bounding Box Dimensions (X, Y, Z)"])

        # Calculate the desired number of points based on the target density
        target_num_points = int(point_density_threshold * bounding_box_volume)

        # Calculate the voxel size such that it achieves the target point density
        voxel_size = ((bounding_box_volume / target_num_points) ** (1/3)) #* 1e-2
        
        return voxel_size
    
    def check_inconsistencies(self):
        """
        This method sequentially performs all the operations from creating masks based on clustering and height analysis,
        finding and filtering inconsistencies, to correcting normals in the point cloud.
        It is a control method that manages the workflow of identifying and correcting inconsistencies.
        """

        # Step 1: Normals based KMeans clustering to create initial masks for blue and black points
        self.normals_kmeans_masks()

        # Step 2: Assign colors and create masks based on height information for red and green points
        self.height_masks()

        # Step 3: Process masks to find unique inconsistencies and combine them
        self.find_inconsistencies()

        # Step 4: Filter out inconsistencies based on cluster size and aspect ratio, and correct the matching points in the original cloud
        self.filter_inconsistencies()

        # Step 5: Correct normals for the points identified as inconsistencies and their neighbors
        self.correct_normals()

        print("Inconsistency check and correction completed.")

    
    def normals_kmeans_masks(self, num_clusters=4):
        """
        This method performs KMeans clustering on the normals of the point cloud.
        It then assigns colors based on the cluster and identifies the clusters
        with the highest and lowest z values, assigning them blue and black colors respectively.

        Args:
            num_clusters (int): The number of clusters to find within the point cloud normals.

        Returns:
            (list, list): Two lists containing the indices of points that belong to the clusters with
            the maximum and minimum z values respectively.
        """
        # Ensure normals are present
        if not self.input_cloud.has_normals():
            print("Point cloud has no normals. Please compute normals before clustering.")
            return [], []

        # Perform KMeans clustering on normals
        normals = np.asarray(self.input_cloud.normals)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(normals)
        labels = kmeans.labels_

        # Initialize variables to track the overall maximum and minimum z values
        max_z = float('-inf')
        min_z = float('inf')
        cluster_with_max_z = None
        cluster_with_min_z = None

        # Group points by clusters and find overall max/min z coordinate for each cluster
        cluster_points = {i: [] for i in range(num_clusters)}  # Dictionary to hold points for each cluster
        for i, label in enumerate(labels):
            z = self.input_cloud.points[i][2]  # Get the z-coordinate of the point
            cluster_points[label].append(z)  # Append z-value to respective cluster

        # Determine the cluster with the max and min z coordinate
        for label, zs in cluster_points.items():
            cluster_max_z = np.max(zs)
            cluster_min_z = np.min(zs)
            if cluster_max_z > max_z:
                max_z = cluster_max_z
                cluster_with_max_z = label
            if cluster_min_z < min_z:
                min_z = cluster_min_z
                cluster_with_min_z = label

        # Create masks for blue and black points
        self.blue_points_mask = []
        self.black_points_mask = []

        # Assign colors and populate masks
        for i, label in enumerate(labels):
            if label == cluster_with_max_z:
                self.blue_points_mask.append(i)
            elif label == cluster_with_min_z:
                self.black_points_mask.append(i)

        return self.blue_points_mask, self.black_points_mask
    
    @staticmethod
    @numba.njit
    def predominant_color_numba(neighbor_colors):
        """
        Static method to determine the predominant color among a set of neighbor colors. The method is executed using numba/JIT compilation to increase performance. 

        Args:
            neighbor_colors (numpy.ndarray): An array of RGB colors of the neighboring points.

        Returns:
            numpy.ndarray: The predominant color in RGB format.
        """
        # Define basic colors in RGB
        red_color = np.array([1, 0, 0])
        green_color = np.array([0, 1, 0])
        white_color = np.array([1, 1, 1])
        black_color = np.array([0, 0, 0])

        red_count, green_count, white_count, black_count = 0, 0, 0, 0

        # Iterate over neighbor_colors and count occurrences of each color
        for i in range(neighbor_colors.shape[0]):
            if np.array_equal(neighbor_colors[i], red_color):
                red_count += 1
            elif np.array_equal(neighbor_colors[i], green_color):
                green_count += 1
            elif np.array_equal(neighbor_colors[i], white_color):
                white_count += 1
            elif np.array_equal(neighbor_colors[i], black_color):
                black_count += 1

        # Find the predominant color
        max_count = max(red_count, green_count, white_count, black_count)
        if max_count == red_count:
            return red_color
        elif max_count == green_count:
            return green_color
        elif max_count == black_count:
            return black_color
        else:
            return white_color
        
    @staticmethod
    @numba.njit
    def process_voxels(points, voxel_size_xy):
        """
        A static method to process a set of points and identify the highest and lowest points (in the z-axis) within each voxel. 
        The method uses voxelization based on the xy-plane to segregate points into different voxels. For each voxel, it 
        determines the indices of points with the maximum and minimum z-values. This method is optimized using Numba's JIT 
        compilation for improved performance.

        Args:
            points (numpy.ndarray): An array of points in the point cloud.
            voxel_size_xy (float): The size of the voxel in the xy-plane.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Two arrays containing indices of points with maximum and minimum z-values 
            in each voxel, respectively.
    """
        # Initialize min and max voxel indices
        min_voxel_idx = np.array([np.inf, np.inf], dtype=np.float64)
        max_voxel_idx = np.array([-np.inf, -np.inf], dtype=np.float64)

        # Determine the range of voxel indices: this loop goes through each point and calculates its voxel index based on its x and y coordinates.
        for point in points:
            voxel_idx = point[:2] // voxel_size_xy
            for i in range(2):  # Loop over x and y dimensions
                # Update the min and max voxel indices for each dimension
                if voxel_idx[i] < min_voxel_idx[i]:
                    min_voxel_idx[i] = voxel_idx[i]
                if voxel_idx[i] > max_voxel_idx[i]:
                    max_voxel_idx[i] = voxel_idx[i]

        # Convert the min and max voxel indices to integer type
        min_voxel_idx = min_voxel_idx.astype(np.int64)
        max_voxel_idx = max_voxel_idx.astype(np.int64)
        # Calculate the total range of voxel indices in each dimension
        voxel_range = max_voxel_idx - min_voxel_idx + 1

        # Initialize arrays to store max and min z values and their indices for each voxel
        max_z_values = np.full((voxel_range[0], voxel_range[1]), -np.inf)
        min_z_values = np.full((voxel_range[0], voxel_range[1]), np.inf)
        max_z_indices = np.full((voxel_range[0], voxel_range[1]), -1, dtype=np.int64)
        min_z_indices = np.full((voxel_range[0], voxel_range[1]), -1, dtype=np.int64)

        # Process each point to update max and min z values and their indices
        for i in range(len(points)):
            # Calculate the voxel index for the current point
            voxel_idx = (points[i, :2] // voxel_size_xy).astype(np.int64) - min_voxel_idx
            # Update the max z value and its index if the current point's z value is higher than the stored max z value
            if points[i, 2] > max_z_values[voxel_idx[0], voxel_idx[1]]:
                max_z_values[voxel_idx[0], voxel_idx[1]] = points[i, 2]
                max_z_indices[voxel_idx[0], voxel_idx[1]] = i
            # Update the min z value and its index if the current point's z value is lower than the stored min z value
            if points[i, 2] < min_z_values[voxel_idx[0], voxel_idx[1]]:
                min_z_values[voxel_idx[0], voxel_idx[1]] = points[i, 2]
                min_z_indices[voxel_idx[0], voxel_idx[1]] = i

        return max_z_indices, min_z_indices
    
    def height_masks(self, voxel_size_xy=0.5, n_neighbors=128, batch_size=25000):
        """
            Method to assign colors to points based on their z-values and the predominant color of their neighbors.

            Args:
                voxel_size_xy (float): The size of the voxel in the xy-plane for grouping points.
                n_neighbors (int): The number of neighbors to consider for determining the predominant color.

            Returns:
                (list, list): Two lists containing indices of red and green points respectively.
        """
        # obtain points and initialize color array
        points = np.asarray(self.input_cloud.points)
        colors = np.ones((len(points), 3))  # Default color is white for all points

        # Process voxels using Numba
        max_z_indices, min_z_indices = self.process_voxels(points, voxel_size_xy)

        # Assign colors based on max and min z values
        for i in range(max_z_indices.shape[0]):
            for j in range(max_z_indices.shape[1]):
                if max_z_indices[i, j] != -1:
                    colors[max_z_indices[i, j]] = [1, 0, 0]  # Red
                if min_z_indices[i, j] != -1:
                    colors[min_z_indices[i, j]] = [0, 1, 0]  # Green
                if max_z_indices[i, j] != -1 and min_z_indices[i, j] != -1 and max_z_indices[i, j] == min_z_indices[i, j]:
                    colors[max_z_indices[i, j]] = [0, 0, 0]  # Black

        # Use NearestNeighbors to find the k-nearest neighbors for each point
        knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(points)
        
        # Initialize corrected colors
        corrected_colors = colors.copy()

        # Process k-nearest neighbors in batches
        for batch_start in range(0, len(points), batch_size):
            batch_end = min(batch_start + batch_size, len(points))
            _, indices = knn.kneighbors(points[batch_start:batch_end])

            # Correct the color of each point in the batch based on the predominant color of its neighbors
            for i, neighbors in enumerate(indices):
                neighbor_colors = colors[neighbors]  # Get the colors of the neighbors
                corrected_colors[batch_start + i] = self.predominant_color_numba(neighbor_colors)

        # Use NearestNeighbors to find the k-nearest neighbors for non-black points
        non_black_indices = np.where(np.any(corrected_colors != [0, 0, 0], axis=1))[0]
        non_black_points = points[non_black_indices]
        non_black_knn = NearestNeighbors(n_neighbors=12, algorithm='auto').fit(non_black_points)

        # Correct black points by examining the neighborhood in batches
        black_indices = np.where(np.all(corrected_colors == [0, 0, 0], axis=1))[0]
        for batch_start in range(0, len(black_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(black_indices))
            black_indices_batch = black_indices[batch_start:batch_end]

            # Find neighbors for each black point in the batch
            _, neighbors_batch = non_black_knn.kneighbors(points[black_indices_batch])

            # Correct the color of each black point based on the predominant color of its non-black neighbors
            for i, neighbors in enumerate(neighbors_batch):
                neighbor_colors = corrected_colors[non_black_indices[neighbors]]
                corrected_colors[black_indices_batch[i]] = self.predominant_color_numba(neighbor_colors)

        # Create masks based on the corrected colors for red and green
        self.red_points_mask = [i for i, color in enumerate(corrected_colors) if np.array_equal(color, [1, 0, 0])]
        self.green_points_mask = [i for i, color in enumerate(corrected_colors) if np.array_equal(color, [0, 1, 0])]

        return self.red_points_mask, self.green_points_mask
    
    def find_inconsistencies(self):
        """
        Processes the color masks to identify points uniquely belonging to certain color categories and combines these to create a subset of the original point cloud.

        This method assumes that masks for blue, black, red, and green points (self.blue_points_mask, self.black_points_mask, self.red_points_mask, and self.green_points_mask) have been previously defined and contain indices for the original point cloud 'self.input_cloud'.

        Returns:
            list: The combined indices representing the unique set of points after processing the masks.
        """
        
        # Convert masks from list to set for efficient set operations
        blue_set = set(self.blue_points_mask)
        red_set = set(self.red_points_mask)
        green_set = set(self.green_points_mask)
        black_set = set(self.black_points_mask)

        # Process blue and red masks
        # Blue points that are not red and vice versa
        blue_not_red_indices = list(blue_set - red_set)
        red_not_blue_indices = list(red_set - blue_set)

        # Process green and black masks
        # Green points that are not black and vice versa
        green_not_black_indices = list(green_set - black_set)
        black_not_green_indices = list(black_set - green_set)

        # Combine sets for intersections and unions
        # Find common indices between blue-not-red and black-not-green sets
        # Similarly, find common indices between green-not-black and red-not-blue sets
        common_red_black = set(red_not_blue_indices) & set(black_not_green_indices)
        common_blue_green = set(blue_not_red_indices) & set(green_not_black_indices)

        # Combine the resulting indices with a union of the two sets
        self.combined_indices = list(common_red_black | common_blue_green)  # Set union

        return self.combined_indices  # Return the combined indices for further processing or analysis
    
    def filter_clusters(self, cluster_sizes, labels, points, size_threshold, aspect_ratio_threshold):
        """
        Filters clusters based on size and aspect ratio.

        Args:
            cluster_sizes (dict): A dictionary of cluster labels and their respective sizes.
            labels (np.array): Labels array from DBSCAN.
            points (np.array): Points corresponding to the combined_indices of the point cloud.
            size_threshold (int): Minimum size of clusters to keep.
            aspect_ratio_threshold (float): Maximum aspect ratio allowed for clusters.

        Returns:
            list: Indices of points that belong to valid clusters.
        """
        large_cluster_labels = [label for label, size in cluster_sizes.items() if size >= size_threshold]
        valid_cluster_indices = []

        for label in large_cluster_labels:
            cluster_mask = labels == label
            cluster_points = points[cluster_mask]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

            obb = cluster_pcd.get_oriented_bounding_box()
            extent = obb.extent
            sorted_extent = sorted(extent, reverse=True)
            aspect_ratio = sorted_extent[0] / sorted_extent[1] if sorted_extent[1] != 0 else np.inf

            if aspect_ratio <= aspect_ratio_threshold:
                valid_cluster_indices.extend(np.where(cluster_mask)[0])

        return valid_cluster_indices

    def find_matching_points(self, filtered_pcd):
        """
        Finds the matching points of a filtered point cloud in the original point cloud.

        Args:
            filtered_pcd (o3d.geometry.PointCloud): The filtered point cloud whose points need to be matched to the original point cloud.

        Returns:
            list: The indices of the original point cloud that correspond to the filtered point cloud.
        """
        pcd_tree = cKDTree(np.asarray(self.input_cloud.points))
        distances, indices = pcd_tree.query(np.asarray(filtered_pcd.points), k=1, distance_upper_bound=1e-5)
        exact_match_indices = indices[distances == 0]
        return exact_match_indices
    
    # def filter_inconsistencies(self, eps_threshold=0.01, size_threshold=1000, aspect_ratio_threshold=3.0):
    def filter_inconsistencies(self, eps_threshold=0.01, size_threshold=500, aspect_ratio_threshold=4.0):
        """
        Filters inconsistencies in the point cloud by clustering points using DBSCAN,
        filtering out small clusters, and ensuring clusters have an acceptable aspect ratio (in order to exclude point cloud edges, which are spurious solutions in the process).
        It then finds the corresponding points in the original point cloud.

        Args:
            combined_indices (list): List of indices from the original point cloud after mask processing.
            eps_threshold (float): The threshold to detect the knee point for epsilon in DBSCAN.
            size_threshold (int): Minimum size of clusters to keep.
            aspect_ratio_threshold (float): Maximum aspect ratio allowed for clusters.

        Returns:
            list: The indices of the original point cloud that correspond to the filtered clusters.
        """

        # Convert combined_indices points to numpy array
        points = np.asarray(self.input_cloud.points)[self.combined_indices]

        # Step 1: Nearest Neighbors for Epsilon Calculation
        neigh = NearestNeighbors(n_neighbors=8)
        nbrs = neigh.fit(points)
        distances, _ = nbrs.kneighbors(points)

        # Step 2: K-distance Graph
        distances = np.sort(distances, axis=0)[:, 1]
        gradient = np.gradient(distances, edge_order=2)
        smoothed_gradient = savgol_filter(gradient, window_length=51, polyorder=3)

        # Step 3: Detecting Knee Point for Epsilon
        exceeds_threshold_indices = np.where(smoothed_gradient[10:] > eps_threshold)[0]
        knee_index = exceeds_threshold_indices[0] + 10 if exceeds_threshold_indices.size > 0 else None
        eps = distances[knee_index] if knee_index is not None else None

        if eps is not None and eps > 0:
            # eps -= 0.2
            # Perform DBSCAN Clustering
            dbscan = DBSCAN(eps=eps, min_samples=3, metric='euclidean')
            labels = dbscan.fit_predict(points)

            # Calculate cluster sizes for non-noise points
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            cluster_sizes = dict(zip(unique_labels, counts))

            # Filter by size and aspect ratio
            valid_cluster_indices = self.filter_clusters(cluster_sizes, labels, points, size_threshold, aspect_ratio_threshold)

            # Create a new point cloud for the filtered clusters
            final_filtered_pcd = o3d.geometry.PointCloud()
            final_filtered_pcd.points = o3d.utility.Vector3dVector(points[valid_cluster_indices])
            
            # Find matching points in the original point cloud
            self.exact_match_indices = self.find_matching_points(final_filtered_pcd)

            return self.exact_match_indices  # Return the matched indices in the original point cloud
        else:
            print("Could not determine a valid 'eps' value from the data")
            return []
        
    def correct_normals(self, k_neighbors=2):
        """
        Corrects normals for the points identified by self.exact_match_indices.
        It inverts the normals of these points, searches for their neighbors to correct similar inconsistencies,
        paints the corrected points red, and visualizes the point cloud.

        Args:
            k_neighbors (int): The number of neighbors to search for each point.
        """

        # Ensure normals are available
        if not self.input_cloud.has_normals():
            print("Normals are not available in the input cloud. Please compute them first.")
            return

        # Invert the normals for the points with indices in self.exact_match_indices
        for idx in self.exact_match_indices:
            # Inverting normal by multiplying by -1
            self.input_cloud.normals[idx] = -1 * np.asarray(self.input_cloud.normals[idx])

        # Initialize KDTree for neighbor search
        pcd_tree = cKDTree(np.asarray(self.input_cloud.points))

        # Find neighbors of each point in self.exact_match_indices and invert their normals if needed
        for idx in self.exact_match_indices:
            point = self.input_cloud.points[idx]
            _ , indices = pcd_tree.query(point, k=k_neighbors + 1)  # +1 because the point itself is included
            for i, neighbor_idx in enumerate(indices):
                if neighbor_idx in self.exact_match_indices or neighbor_idx == idx:
                    # Skip if it's the point itself or already corrected
                    continue
                # Check if the normal is in the opposite direction and invert it
                # Here we use np.any to check if any component's sign is different
                if np.any(np.sign(self.input_cloud.normals[idx]) != np.sign(self.input_cloud.normals[neighbor_idx])):
                    # Inverting normal by multiplying by -1
                    self.input_cloud.normals[neighbor_idx] = -1 * np.asarray(self.input_cloud.normals[neighbor_idx])

        # Clone the input cloud for visualization purposes using deepcopy
        display_cloud = copy.deepcopy(self.input_cloud)

        # Paint the corrected points red in the cloned cloud
        for idx in self.exact_match_indices:
            display_cloud.colors[idx] = [1, 0, 0]  # Red color for corrected points

        # Visualize the cloned point cloud with modified normals
        print("Displaying the inconsistent points with corrected normals in red.")
        o3d.visualization.draw_geometries([display_cloud])

    def remove_external_structures(self, file_path):
        """
        Remove external structures from the point cloud based on a provided .ply file
        containing the 'leftovers' to be removed.

        Args:
            file_path (str): Path to the .ply file containing the main point cloud data.

        Returns:
            o3d.geometry.PointCloud: The cleaned point cloud with external structures removed.
        """

        # Construct the path to the leftovers file
        base_file_path, _ = os.path.splitext(file_path)
        leftovers_file_path = base_file_path + "_leftovers.ply"

        if not os.path.exists(leftovers_file_path):
            print(f"Leftovers file not found: {leftovers_file_path}")
            print("Please perform external structures detection on CloudCompare and load the .ply in the current file folder")
            return self.input_cloud

        # Load the leftovers point cloud
        leftovers_pcd = o3d.io.read_point_cloud(leftovers_file_path)

        # Build a KDTree for the main cloud
        kdtree = o3d.geometry.KDTreeFlann(self.input_cloud)

        # Loop through leftovers and find closest points in the main cloud
        indices_to_remove = []
        for point in leftovers_pcd.points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            if idx:  # Check if the list is not empty
                indices_to_remove.append(idx[0])

        # Remove the points from the main cloud
        self.input_cloud = self.input_cloud.select_by_index(indices_to_remove, invert=True)

        print("External structures detected using RANSAC Shape Detection (CloudCompare) removed from running point cloud")

        return self.input_cloud

    

