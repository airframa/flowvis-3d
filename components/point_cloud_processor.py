import os
import open3d as o3d
import cv2
import numpy as np
import numba
from numba import jit
import copy
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon
from skimage import morphology, draw
import alphashape
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor
import multiprocessing 
from point_cloud_utils import calculate_point_cloud_metrics
from point_cloud_visualizer import PointCloudVisualizer
import random


class PointCloudProcessor():
    def __init__(self, file_path, input_cloud, full_res_pcd):
        """
        Initialize the PointCloudProcessor for processing point clouds and extracting flow visualization lines. The class is intended as a broad toolbox for point cloud texture data processing. 

        Args:
            source (str or o3d.geometry.PointCloud): Either the path to a PLY file containing the point cloud or the
            point cloud itself. 
            
        This class is designed to provide tools for processing and visualizing point cloud data with the focus on extracting flow-vis lines from the point cloud
        """
        # store file path in globally available attribute
        self.file_path = file_path
        # We retain copies of the input cloud to allow for multiple class instances in the main.ipynb without modifying the input cloud (experimentation phase)
        if isinstance(input_cloud, str):
            self.input_cloud = copy.deepcopy(o3d.io.read_point_cloud(input_cloud))      
            # store original downsampled point cloud for final display
            self.input_cloud_copy = copy.deepcopy(o3d.io.read_point_cloud(input_cloud))     
        elif isinstance(input_cloud, o3d.geometry.PointCloud):
            self.input_cloud = copy.deepcopy(input_cloud)
            # store original downsampled point cloud for final display
            self.input_cloud_copy = copy.deepcopy(input_cloud)
        if isinstance(full_res_pcd, str):
            self.full_res_pcd = copy.deepcopy(o3d.io.read_point_cloud(full_res_pcd)) 
            # store original full resolution point cloud for final display
            self.full_res_pcd_copy = copy.deepcopy(full_res_pcd)         
        elif isinstance(full_res_pcd, o3d.geometry.PointCloud):
            self.full_res_pcd = copy.deepcopy(full_res_pcd)
            # store original full resolution point cloud for final display
            self.full_res_pcd_copy = copy.deepcopy(full_res_pcd)
        else:
            raise ValueError("Invalid source type. Provide a path to a PLY file or an existing point cloud.")
        # Variables Initialization
        # PCD SEGMENTATION # 
        # Initialize the segments list where the pcd segments will be stored 
        self.segments = []
        # Initialize the rotated segments list where the pcd segments with PCA rotation will be stored 
        self.segments_pca = []
        # Initialize the clusters list where the pcd segments will be stored 
        self.segment_features = []
        # Initialize the folder where the results of the 2D analysis are stored
        self.output_folder = None
        # Initialize the list to store the segment data (this step is needed to ensure a smooth 3D-2D-3D transition when dealing with segments)
        self.segment_data_2d = {}
        # 2D IMAGE ANALYSIS
        self.c = None # adaptive thresholding constant
        self.size_threshold = None # threshold for small component filtering operation
        # ADAPTIVE METHODS #
        # Initialize the clusters labels (obtained through DBSCAN) in the context of the adaptive methods implementation.  The labels are needed during point cloud processing to access each single flowvis cluster separately
        self.labels = []  # Initialize labels list
        
        
    ##################################################################################################################################################################################
    # STATIC PRE-PROCESSING
    ##################################################################################################################################################################################

    def detect_flowvis_color(self):
        """
        Detect the predominant color in the point cloud using uniform sampling and k-means clustering.

        Args:
        Output: The method utilizes the internal state of the class ('input_cloud' attribute).

        """
        # Extract color information
        colors = np.asarray(self.input_cloud.colors)
        
        ## Apply k-means clustering to sampled colors
        num_clusters = 2
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(colors)
        # Calculate the average color intensity for each cluster
        cluster_avg_colors = []

        for label in range(num_clusters):
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_avg_color = np.mean(colors[cluster_indices], axis=0)
            cluster_avg_colors.append(np.sum(cluster_avg_color))

        # Determine the cluster with the highest average color intensity (brightest colors)
        bright_cluster = np.argmax(cluster_avg_colors)
        # Get the indices of points in the brightest cluster
        bright_cluster_indices = np.where(cluster_labels == bright_cluster)[0]
        # Calculate the average color of the brightest cluster
        bright_cluster_avgcolor = np.mean(colors[bright_cluster_indices], axis=0) * 255
        # Extract RGB coordinates
        r, g, b = bright_cluster_avgcolor
        
        # Assign a label based on the predominant cluster
        if np.all(np.array([r/g, b/g]) < 0.65):
            self.flowvis_color = "green"
        else:
            self.flowvis_color = "white"

        print(f"Flow-vis color : {self.flowvis_color}")
          
    
    def filter_by_color_ranges(self, color_ranges, keep=True):
        """
        Filter points in the point cloud based on specified, user-defined, color ranges. The color range provided as input can be retained or discarded. 

        Args:
            color_ranges (list): List of tuples specifying lower and upper RGB color bounds.
            keep (bool, optional): If True, retain points within the specified color ranges.
                                If False, discard points within the specified color ranges. Default is True.
        """
        if self.input_cloud is None:
            raise ValueError("No point cloud available. Load a point cloud first.")

        # Extract the RGB colors from the point cloud
        colors = np.asarray(self.input_cloud.colors) * 255

        # Create a mask based on color ranges
        mask = np.zeros(len(colors), dtype=bool)
        for color_range in color_ranges:
            lower, upper = color_range

            # Check if each color component is within the specified range
            in_range = np.logical_and.reduce(
                (colors[:, 0] >= lower[0], colors[:, 1] >= lower[1], colors[:, 2] >= lower[2],
                colors[:, 0] <= upper[0], colors[:, 1] <= upper[1], colors[:, 2] <= upper[2])
            )

            # Combine the individual color range masks using logical OR
            mask = np.logical_or(mask, in_range)

        if keep:
            # Retain points in the specified color ranges
            self.input_cloud = self.input_cloud.select_by_index(np.where(mask)[0])
        else:
            # Discard points in the specified color ranges
            self.input_cloud = self.input_cloud.select_by_index(np.where(~mask)[0])

        # Visualize the filtered point cloud (optional)
        # o3d.visualization.draw_geometries([self.input_cloud])

    def enhance_contrast(self, contrast_factor):
        """
        Enhance contrast in the point cloud while preserving original colors.

        Args:
            contrast_factor (float): The contrast factor.
            pcd (o3d.geometry.PointCloud): The input point cloud.

        Returns:
            o3d.geometry.PointCloud: The enhanced point cloud.
        """
        if self.input_cloud is None:
            raise ValueError("No point cloud available. Load a point cloud first.")

        # Extract the RGB colors from the point cloud
        rgb_colors = np.asarray(self.input_cloud.colors) * 255
        # Apply contrast enhancement to each color channel separately
        enhanced_colors = (rgb_colors - 128) * contrast_factor + 128
        # Clip pixel values to be within the [0, 255] range
        enhanced_colors = np.clip(enhanced_colors, 0, 255)
        # Create a new PointCloud with the enhanced RGB colors
        self.input_cloud.colors = o3d.utility.Vector3dVector(enhanced_colors / 255.0)

        # Visualize the enhanced contrast point cloud
        # o3d.visualization.draw_geometries([self.input_cloud])


    def convert_to_grayscale(self):
        """
        Convert the point cloud from RGB to grayscale.

        Returns:
            None
        """
        if self.input_cloud is None:
            raise ValueError("No point cloud available. Load a point cloud first.")
        
        # Compute point normals for visualization purposes
        # radius_normal = 1.0 
        # self.input_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # Extract RGB colors from the point cloud
        rgb_colors = np.asarray(self.input_cloud.colors)

        # Convert RGB colors to grayscale
        grayscale = np.dot(rgb_colors, [0.2989, 0.5870, 0.1140])  # Use the standard grayscale conversion weights

        # gaussian blur/median blur for adaptive threshold: rejected, only for documentation purposes
        #grayscale = median_filter(grayscale, size=2)

        # Update the input point cloud with grayscale colors
        self.input_cloud.colors = o3d.utility.Vector3dVector(np.stack([grayscale, grayscale, grayscale], axis=-1))

        # Visualize the grayscale point cloud
        # o3d.visualization.draw_geometries([self.input_cloud])

    def threshold_grayscale(self, threshold):
        """
        Threshold the grayscale point cloud to assign black color to points below the threshold and white color to points
        above the threshold.

        Args:
            threshold (float): The threshold value to separate black and white regions.

        Returns:
            None
        """
        if self.input_cloud is None:
            raise ValueError("No grayscale point cloud available. Convert the point cloud to grayscale first.")

        # Extract grayscale values from the grayscale point cloud
        grayscale_values = np.asarray(self.input_cloud.colors)[:, 0] * 255
        # Create a mask for points below the threshold
        below_threshold_mask = grayscale_values < threshold
        # Convert the Open3D colors to a numpy array
        colors = np.asarray(self.input_cloud.colors)
        # Set black color (all zeros) for points below the threshold
        colors[below_threshold_mask] = [0.0, 0.0, 0.0]
        # Set white color (all ones) for points above the threshold
        colors[~below_threshold_mask] = [1.0, 1.0, 1.0]
        # Update the input point cloud with the modified colors
        self.input_cloud.colors = o3d.utility.Vector3dVector(colors)
        # retain white flow-vis paint points in a separate point cloud        
        # white_points_cloud = self.input_cloud.select_by_index(np.where(~below_threshold_mask)[0])

        # Visualize the result with white points in red and "black" points with original colors
        # white_points_cloud_copy = copy.deepcopy(white_points_cloud)
        # white_points_cloud_copy.paint_uniform_color([1.0, 0.0, 0.0])
        # o3d.visualization.draw_geometries([white_points_cloud_copy, self.input_cloud_copy])

    def kmeans_color_clustering(self, num_clusters=2):
        """
        Apply K-Means clustering based on colors to the point cloud.

        Args:
            num_clusters (int): The number of clusters to create.

        Returns:
            o3d.geometry.PointCloud: The clustered point cloud.
        """
        if self.input_cloud is None:
            raise ValueError("No point cloud available. Load a point cloud first.")
        
        # Compute point normals for visualization purposes
        radius_normal = 1.0 
        self.input_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        # Extract the RGB colors from the point cloud
        rgb_colors = np.asarray(self.input_cloud.colors) * 255
        # Apply K-Means clustering to colors
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(rgb_colors)
        # Create a list of unique cluster labels
        unique_labels = np.unique(cluster_labels)

        # Create a new point cloud for each cluster
        clustered_point_clouds = []
        # clustered_point_clouds_copy = []

        for label in unique_labels:
            # Get the indices of points in this cluster
            cluster_indices = np.where(cluster_labels == label)[0]
            # Select points of this cluster
            cluster_points = self.input_cloud.select_by_index(cluster_indices)
            # append cluster
            clustered_point_clouds.append(cluster_points)
            
            # visualization
            # Create a new point cloud with original RGB colors for visualization purposes
            # cluster_points_original = o3d.geometry.PointCloud()
            # cluster_points_original.points = cluster_points.points
            # original_colors = rgb_colors[cluster_indices] / 255.0
            # cluster_points_original.colors = o3d.utility.Vector3dVector(original_colors)
            # Visualize the cluster with original RGB colors
            # o3d.visualization.draw_geometries([cluster_points_original])
            # provide cluster operation visualization
            # Create a deep copy of the point cloud
            # cluster_points_copy = copy.deepcopy(cluster_points)
            # Set a uniform color for the clusters: black for non-flow vis, white for flow vis
            # if label == 0:
                #cluster_points_copy.paint_uniform_color([0, 0, 0])  # Black
            # else:
                # cluster_points_copy.paint_uniform_color([1, 1, 1]) 
            # append cluster for final representation
            # clustered_point_clouds_copy.append(cluster_points_copy)

        # Visualize the clustered point clouds - cluster represented in different colors
        # o3d.visualization.draw_geometries(clustered_point_clouds_copy)

        # Visualize original point cloud with detected flow-vis represented in red
        # clustered_point_clouds_copy[1].paint_uniform_color([1.0, 0.0, 0.0]) 
        # o3d.visualization.draw_geometries([clustered_point_clouds_copy[1], self.input_cloud_copy])

        self.input_cloud = clustered_point_clouds

    ##################################################################################################################################################################################
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # FLOW-VIS DETECTION AND CLASSIFICATION - 2D APPROACH
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##################################################################################################################################################################################

    ##################################################################################################################################################################################
    # SURFACE DETECTION AND POINT CLOUD SEGMENTATION METHODS
    ################################################################################################################################################################################## 
    
    def kmeans_normals_clustering(self, num_clusters):
        """
        Apply K-Means clustering to the point cloud based on point cloud normals.

        Args:
            num_clusters (int): Number of clusters for K-Means.

        Returns:
            list: List of clustered point clouds.
        """
        if self.input_cloud is None:
            raise ValueError("No point cloud available. Load a point cloud first.")

        # Check if normals are already computed
        if not self.input_cloud.has_normals():
            # Normals are not computed, so compute them
            radius_normal = 2.0
            self.input_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            # Orient the normals with respect to consistent tangent planes
            self.input_cloud.orient_normals_consistent_tangent_plane(k=10)

        # Extract normals as a numpy array
        normals = np.asarray(self.input_cloud.normals)
        # Apply K-Means clustering to normals
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(normals)
        # Create a list of unique cluster labels
        unique_labels = np.unique(cluster_labels)
        # Create a list of clustered point clouds
        clustered_point_clouds = []
        clustered_point_clouds_copy = []

        # Visualize the clustered point clouds with random colors
        for label in unique_labels:

            cluster_indices = np.where(cluster_labels == label)[0]
            cluster = self.input_cloud.select_by_index(cluster_indices)
            # Create a deep copy of the cluster point cloud
            cluster_copy = copy.deepcopy(cluster)
            # Set a random color for the cluster for representation purposes
            cluster_copy.paint_uniform_color(list(np.random.rand(3)))
            # append clusters
            clustered_point_clouds.append(cluster)
            clustered_point_clouds_copy.append(cluster_copy)

        # Visualize the clustered point clouds
        # o3d.visualization.draw_geometries(clustered_point_clouds_copy)

        return clustered_point_clouds, cluster_labels

    def calculate_optimal_epsilon(self, points, start_index=5000, threshold=0.00005, n_neighbors=8, plot=False):
        """
        Static method to calculate the optimal epsilon value for DBSCAN clustering.
        It applies the K-distance method to determine the knee point which serves as the optimal epsilon.

        Args:
            points (numpy.ndarray): The array of point cloud data.
            start_index (int): The index to start searching for the knee point in the sorted distances.
            threshold (float): The threshold value to detect the knee point in the gradient of distances.
            n_neighbors (int): The number of nearest neighbors to consider for each point.
            plot (bool): If True, plot the K-distance graph with the detected knee point.

        Returns:
            float: The optimal epsilon value for DBSCAN clustering.
        """
        # Step 1: Nearest Neighbors for Epsilon Calculation
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs = neigh.fit(points)
        distances, _ = nbrs.kneighbors(points)

        # Step 2: K-distance Graph
        distances = np.sort(distances, axis=0)[:, 1]
        gradient = np.gradient(distances, edge_order=2)
        smoothed_gradient = savgol_filter(gradient, window_length=51, polyorder=3)

        # Step 3: Detecting Knee Point for Epsilon
        exceeds_threshold_indices = np.where(smoothed_gradient[start_index:] > threshold)[0]
        knee_index = exceeds_threshold_indices[0] + start_index if exceeds_threshold_indices.size > 0 else None
        eps = distances[knee_index] if knee_index is not None else None

        # Step 4: Plot the K-distance graph with the detected point
        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(distances, label='K-distance')
            if knee_index is not None:
                plt.axvline(x=knee_index, color='red', linestyle='--', label=f'Knee Point (eps = {eps})')
            plt.legend()
            plt.xlabel('Data Points sorted by distance')
            plt.ylabel('Epsilon')
            plt.title('K-distance Graph with Detected Knee Point')
            plt.show()

        return eps
        
    def segment_pcd(self, manifold=True, simple_surface=False, eps_additional=0.25, min_cluster_size=2000, plot_eps_opt_dbscan=False):
        """
        Segments the point cloud into different surfaces based on normal vectors and spatial clustering. It uses K-Means to segment
        the point cloud based on normals and then applies DBSCAN for spatial clustering on each segment.

        Args:
            manifold (bool): If true, the point cloud is assumed to have a more complex structure with multiple-sided surfaces, i.e. closed volumes with a upper and lower surface. This is the case, for example, of front wing flaps, suspension links, rear wing main planes (if both sides are captured). If False, then it 
            is assumed that these objects are either not present or not captured (e.g., Rear Wing main plane scan where only the internal surface is captured).
            simple_surface (bool): If true, it indicates the presence of a simple surface (e.g. single RW main plane, i.e. set of surfaces displaying only one main direction). In this case, K-means based on pcd normals is skipped and DBSCAN is directly performed. 
            eps_additional (float): Additional epsilon to be considered for DBSCAN to adjust cluster sensitivity.
            min_cluster_size (int): Minimum size for a cluster to be considered valid after DBSCAN clustering.
            plot_eps_opt_dbscan (bool): flag to display (or not) the optimal epsilon plot (K-distance neighbours knee point) in the context of DBSCAN clustering

        Returns:
            list: A list of point cloud segments after K-means and DBSCAN clustering, each segment as an individual point cloud.
        """
        if self.input_cloud is None:
            raise ValueError("No point cloud available. Load a point cloud first.")

        # Compute normals if they are not already present
        if not self.input_cloud.has_normals():
            radius_normal = 2.0
            self.input_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            # Orient the normals with respect to consistent tangent planes
            self.input_cloud.orient_normals_consistent_tangent_plane(k=10)

        # select the number of clusters for K-means
        # if the point cloud at hand is manifold (i.e. upper and lower surfaces are present), then num_clusters = 4 in order to separate lower and upper surfaces, otherwise num_clusters = 3. In case of complex surfaces (e.g., diffuser) the clusters are 5 and then a merging operation is carried out. In case of a simple surface,
        # such as the main-plane of a Rear Wing, K-means clustering is skipped altogether
        if manifold:
            num_clusters_kmeans = 4
        else:
            num_clusters_kmeans = 3

        # Apply K-Means clustering to normals
        if not simple_surface:
            # Existing K-means and DBSCAN logic
            _ , cluster_labels_kmeans = self.kmeans_normals_clustering(num_clusters_kmeans)
            # Create a list of unique cluster labels
            unique_labels_kmeans = np.unique(cluster_labels_kmeans)
        else:
            # Artificially assign the same label to all points when simple_surface is True
            print("Skipping K-means clustering step due to simple surface condition")
            cluster_labels_kmeans = np.zeros(len(self.input_cloud.points)) # Assign 0 as the label for all points
            unique_labels_kmeans = np.unique(cluster_labels_kmeans)

        # Informing user about the segmentation using K-Means
        print(f"Point cloud segmented into {len(unique_labels_kmeans)} clusters using K-means applied on the point cloud normal vectors")
        # Create a list of clustered point clouds for DBSCAN clustering operation for display purposes
        clusters_dbscan_display = []
        
        # Process each cluster identified by K-Means
        for i, label_kmeans in enumerate(unique_labels_kmeans):
            # Inform user which cluster is currently being processed
            print(f"Processing cluster {i+1} from K-means with DBSCAN")
            # Extract individual cluster from K-means clustering operation
            cluster_indices = np.where(cluster_labels_kmeans == label_kmeans)[0]
            cluster = self.input_cloud.select_by_index(cluster_indices)
            # scalculate optimal epsilon value for DBSCAN
            eps = self.calculate_optimal_epsilon(cluster.points, start_index=5000, threshold=0.00005, n_neighbors=8, plot=plot_eps_opt_dbscan)

            # Step 4: DBSCAN Clustering
            if eps is not None:
                min_samples=3
                eps += eps_additional  # more or less aggregative clustering strategy
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                labels_dbscan = dbscan.fit_predict(np.asarray(cluster.points))

                # Count the points in each cluster and filter out small clusters and noise (-1 label in DBSCAN)
                cluster_indices = [np.where(labels_dbscan == i)[0] for i in range(max(labels_dbscan)+1)]
                cluster_sizes = [len(ci) for ci in cluster_indices]
                valid_clusters = [ci for ci, size in zip(cluster_indices, cluster_sizes) if size >= min_cluster_size and size != -1]

                # Informing user about the number of DBSCAN clusters after filtering
                print(f"Number of DBSCAN clusters before filtering: {len(cluster_indices)}")
                print(f"Number of DBSCAN clusters after filtering: {len(valid_clusters)}")

                for ci in valid_clusters:
                    cluster_points = np.asarray(cluster.points)[ci]
                    cluster_normals = np.asarray(cluster.normals)[ci]
                    cluster_colors = np.asarray(cluster.colors)[ci]  # Assuming cluster.colors is aligned with cluster.points

                    # Create a new point cloud object for the cluster
                    cluster_pcd = o3d.geometry.PointCloud()
                    cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                    cluster_pcd.normals = o3d.utility.Vector3dVector(cluster_normals)
                    cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

                    # Creata a copy of each cluster for visualization purposes and assign a random color
                    cluster_dbscan_display = copy.deepcopy(cluster_pcd)
                    random_color = list(np.random.rand(3))  # Generate a random color
                    cluster_dbscan_display.paint_uniform_color(random_color)  # Apply color to the cluster

                    # Append the new cluster to the list of clusters
                    self.segments.append(cluster_pcd)
                    clusters_dbscan_display.append(cluster_dbscan_display)

            else:
                print("No knee point detected with the given threshold.")
                return None

        # Final statement after all clustering is done
        print(f"Initial point cloud segmented into {len(self.segments)} clusters, representing the point cloud different surfaces")

        # Finally, visualize all clusters together
        print(f"Displaying all {len(self.segments)} clusters with random colors\n")
        # Instantiate visualizer to display the segmentation result
        print("Segmented point cloud:")
        PointCloudVisualizer(clusters_dbscan_display)

        return  self.segments
        
    def retrieve_corresponding_region(self, segment_cloud, eps_additional):
        """
        Retrieves and filters the region corresponding to a segment of the point cloud by iteratively applying a voxel grid filter
        with pre-defined voxel sizes, followed by DBSCAN clustering to remove smaller clusters.

        Parameters:
        segment_cloud (open3d.geometry.PointCloud): The segment of the point cloud to refine.
        full_resolution_pcd (open3d.geometry.PointCloud): The full resolution point cloud.

        Returns:
        open3d.geometry.PointCloud: The filtered corresponding region of the point cloud.
        List[int]: The indices of the filtered points in the full resolution point cloud.
        """
        # voxel_sizes = [1.0, 0.8, 1.0, 0.5, 0.6, 1.0, 0.8, 1.0]
        voxel_sizes = [1.0, 0.8, 1.0, 0.6, 1.0, 0.5]

        for voxel_size in voxel_sizes:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(segment_cloud, voxel_size)
            original_points = np.asarray(self.full_res_pcd.points)
            original_points_o3d = o3d.utility.Vector3dVector(original_points)
            included_mask = voxel_grid.check_if_included(original_points_o3d)

            included_indices = np.where(included_mask)[0]
            included_points = original_points[included_indices]
            if self.full_res_pcd.has_colors():
                original_colors = np.asarray(self.full_res_pcd.colors)
                included_colors = original_colors[included_indices]

            corresponding_region = o3d.geometry.PointCloud()
            corresponding_region.points = o3d.utility.Vector3dVector(included_points)
            if self.full_res_pcd.has_colors():
                corresponding_region.colors = o3d.utility.Vector3dVector(included_colors)
            segment_cloud = corresponding_region

        # DBSCAN Filtering Logic
        eps = self.calculate_optimal_epsilon(included_points)
        # eps = None
        if eps is not None:
            eps += eps_additional  # More aggregative clustering strategy
            dbscan = DBSCAN(eps=eps, min_samples=3, metric='euclidean')
            labels_dbscan = dbscan.fit_predict(included_points)
            unique_labels, counts = np.unique(labels_dbscan, return_counts=True)

            # Filtering clusters with at least 10,000 points and excluding noise
            large_clusters = unique_labels[(counts >= 10000) & (unique_labels != -1)]

            # Collecting indices of points in large clusters
            filtered_indices = np.array([], dtype=int)
            for label in large_clusters:
                label_indices = np.where(labels_dbscan == label)[0]
                filtered_indices = np.concatenate((filtered_indices, included_indices[label_indices]))

            if filtered_indices.size == 0:
                return o3d.geometry.PointCloud(), np.array([])  # Return empty if no large clusters found

            # Filtering points and colors for the large clusters
            filtered_points = included_points[np.isin(labels_dbscan, large_clusters)]
            filtered_colors = included_colors[np.isin(labels_dbscan, large_clusters)] if self.full_res_pcd.has_colors() else None

            # Creating the final filtered point cloud
            filtered_region = o3d.geometry.PointCloud()
            filtered_region.points = o3d.utility.Vector3dVector(filtered_points)
            if filtered_colors is not None:
                filtered_region.colors = o3d.utility.Vector3dVector(filtered_colors)

            return filtered_region, filtered_indices

        else:
            return corresponding_region, included_indices

        
    def process_single_cluster_parallel(self, segment_cloud, eps_additional):
        """
        Process a single cluster by finding its corresponding region and then filtering it.

        Parameters:
        segment_cloud (open3d.geometry.PointCloud): The segment of the point cloud to process.
        full_resolution_pcd (open3d.geometry.PointCloud): The full resolution point cloud.

        Returns:
        open3d.geometry.PointCloud: The filtered point cloud for the cluster.
        """
        filtered_pcd, _ = self.retrieve_corresponding_region(segment_cloud, eps_additional)

        return filtered_pcd
    
    def recover_resolution(self, mode='parallel', optimal=True, eps_additional=0.15):
        """
        Processes each cluster in the segmented_clusters list to retrieve the original point cloud resolution.
        Each cluster is first enhanced by retrieving the original points in the region of interest and then 
        filtered to remove outliers. The segmented_clusters attribute is updated with the new, filtered clusters.
        This method supports both parallel and serial processing modes. The reason for serial processing is in this 
        case the possibility of updating the full resolution point cloud at each iteration, removing the points of
        the segment that was already processed. 
        If user-requested, the optimal mode between 'parallel' and 'series' processing will be used. 

        Parameters:
        full_resolution_pcd (open3d.geometry.PointCloud): The full resolution point cloud.
        mode (str): The mode of operation, either 'parallel' or 'serial'.

        Returns:
        List[open3d.geometry.PointCloud]: A list of filtered point clouds for each cluster.
        """
        n_full_res_points = len(np.asarray(self.full_res_pcd.points)) # number of points in the full resolution pcd
        mode_change = 34  # number of cluster when the switch in most perfromant mode (between 'parallel' and 'series') occurs. this value was dtermined empirically by conducting experiments.
        # If user-requested, use the optimal computational mode between 'parallel' and 'series'
        if optimal:
            if len(self.segments) < mode_change:
                mode = 'parallel'
            else:
                mode = 'series'
            print(f"Optimal mode: {mode}")

        if mode == 'parallel':
            # Using ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor() as executor:
                # Submitting tasks for each cluster to be processed in parallel
                tasks = [executor.submit(self.process_single_cluster_parallel, segment, eps_additional) for segment in self.segments]
                # Waiting for all parallel tasks to complete and collecting the results
                new_clusters = [task.result() for task in tasks]

            # Updating the segmented_clusters attribute with the newly processed clusters
            self.segments = new_clusters

            return self.segments
        
        elif mode == 'serial':
            # Processing each cluster sequentially (serial processing)
            new_clusters = []
            # Copying the full-resolution point cloud's points and colors for manipulation
            updated_full_res_pcd_points = np.asarray(self.full_res_pcd.points)
            updated_full_res_pcd_colors = np.asarray(self.full_res_pcd.colors) if self.full_res_pcd.has_colors() else None

            for segment in self.segments:
                # Finding and filtering the corresponding region for each segment
                filtered_region, filtered_indices = self.retrieve_corresponding_region(segment, eps_additional)
                new_clusters.append(filtered_region)

                # Ensuring indices are integers and within bounds of the updated point cloud
                filtered_indices = filtered_indices.astype(int)
                valid_indices = filtered_indices[filtered_indices < len(updated_full_res_pcd_points)]

                # Removing the processed points from the full-resolution point cloud
                updated_full_res_pcd_points = np.delete(updated_full_res_pcd_points, valid_indices, axis=0)
                if updated_full_res_pcd_colors is not None:
                    updated_full_res_pcd_colors = np.delete(updated_full_res_pcd_colors, valid_indices, axis=0)

                # update the full-resolution point cloud by removing the points which were already processed
                self.full_res_pcd.points = o3d.utility.Vector3dVector(updated_full_res_pcd_points)
                if updated_full_res_pcd_colors is not None:
                    self.full_res_pcd.colors = o3d.utility.Vector3dVector(updated_full_res_pcd_colors)

            # Updating the segmented_clusters attribute with the new clusters
            self.segments = new_clusters

            return self.segments

        else:
            print("Please provide a valid processing mode (mode='parallel' or mode='serial')")

        # provide final statement
        pts_sum = 0
        # find the number of points tht have been rercovered from the full resolution pcd
        for segment in self.segments:
            pts_sum += len(np.asarray(segment.points))
        # calculate recovery ratio
        recovery_ratio = pts_sum/n_full_res_points * 100
        # sometimes a small set of points is assigned to two contiguous segments (rare occasion)
        if recovery_ratio > 100.0:
            recovery_ratio = 100.0
        print(f"Recovery ratio: {recovery_ratio}")

    def pca_pointcloud(self, pcd, segment_idx, flip_axes=[False, False, False]):
        """
        Performs PCA on a point cloud to align it along the principal axes and stores PCA data for later use.

        Args:
            pcd (open3d.geometry.PointCloud): The input point cloud to be processed.
            segment_idx (int): The index of the segment being processed.
            flip_axes (list of bool): Flags to optionally flip the orientation along the principal axes.

        Returns:
            open3d.geometry.PointCloud: The rotated point cloud aligned with the principal axes.
        """

        # Convert point cloud to numpy array and handle colors
        points = np.asarray(pcd.points)
        if pcd.has_colors:
            colors = np.asarray(pcd.colors)

        # Calculate the mean and center the points
        mean = np.mean(points, axis=0)
        centered_points = points - mean

        # Calculate the covariance matrix and perform eigen decomposition
        covariance_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the principal components
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Ensure the third principal component is aligned with the positive z-axis: this is to ensure coherence in the image creation process among the different segments
        if eigenvectors[2, 2] < 0:
            eigenvectors *= -1

        # Rotate the points to align with the principal components
        rotated_points = np.dot(centered_points, eigenvectors)

        # Ensure the dictionary entry for this segment exists
        if segment_idx not in self.segment_data_2d:
            self.segment_data_2d[segment_idx] = {}

        # Store PCA data in segment_data for the specific segment index
        self.segment_data_2d[segment_idx]['mean'] = mean
        self.segment_data_2d[segment_idx]['eigenvectors'] = eigenvectors

        # Create a new point cloud for the rotated points
        rotated_pcd = o3d.geometry.PointCloud()
        rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)
        if pcd.has_colors:
            rotated_pcd.colors = o3d.utility.Vector3dVector(colors)

        return rotated_pcd
    
    def revert_pca_transformation(self, features_pcd, segment_idx):
        """
        Reverts the PCA transformation for a given point cloud segment.

        Args:
            features_pcd (open3d.geometry.PointCloud): The point cloud to be reverted to its original space.
            segment_idx (int): The index of the segment whose PCA transformation is to be reverted.

        Returns:
            open3d.geometry.PointCloud: The point cloud reverted back to the original space.
        """
        # Retrieve the PCA mean and eigenvectors for the segment
        pca_mean = self.segment_data_2d[segment_idx]['mean']
        pca_eigenvectors = self.segment_data_2d[segment_idx]['eigenvectors']

        # Convert the point cloud to a numpy array
        points_pca = np.asarray(features_pcd.points)

        # Revert the PCA transformation: rotate the points back to the original space using the transpose of the eigenvectors matrix and then add the mean
        original_points = np.dot(points_pca, pca_eigenvectors.T) + pca_mean
        # skeleton_3d = np.dot(skeleton_3d_pca, eigenvectors.T) + mean_point

        # Create a new point cloud for the original points
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(original_points)
        if features_pcd.has_colors:
            # Keep the same colors as the input point cloud
            original_pcd.colors = features_pcd.colors

        return original_pcd
    
    @staticmethod
    @numba.njit
    def compute_grid_colors2(points, colors, step_size):
        """
        Computes the colors for each grid cell by averaging the colors of points that fall into the cell.

        Args:
            points (numpy.ndarray): The array of points from the point cloud.
            colors (numpy.ndarray): The array of colors corresponding to the points.
            step_size (float): The step size for grid cell size.

        Returns:
            tuple: A tuple containing the grid colors, grid dimensions, minimum and maximum bounds.
        """
        # Initialize min and max values for each axis
        min_x, min_y, min_z = np.inf, np.inf, np.inf
        max_x, max_y, max_z = -np.inf, -np.inf, -np.inf

        # Compute the min and max values for each axis
        for point in points:
            min_x, min_y, min_z = min(min_x, point[0]), min(min_y, point[1]), min(min_z, point[2])
            max_x, max_y, max_z = max(max_x, point[0]), max(max_y, point[1]), max(max_z, point[2])

        min_bound = np.array([min_x, min_y, min_z], dtype=np.float64)
        max_bound = np.array([max_x, max_y, max_z], dtype=np.float64)

        # Calculate grid dimensions based on step size
        grid_dims = np.ceil((max_bound - min_bound)[:2] / step_size).astype(np.int64)

        # Initialize the grid for storing colors and count of points in each cell
        grid_colors = np.zeros((grid_dims[1], grid_dims[0], 3))
        grid_counts = np.zeros((grid_dims[1], grid_dims[0]))

        # Calculate grid coordinates for each point and update color and count
        for i in range(len(points)):
            grid_x, grid_y = ((points[i, :2] - min_bound[:2]) / step_size).astype(np.int64)
            grid_x, grid_y = min(grid_x, grid_dims[0] - 1), min(grid_y, grid_dims[1] - 1)
            
            grid_colors[grid_y, grid_x] += colors[i] * 255
            grid_counts[grid_y, grid_x] += 1

        # # Manually average the colors in each grid cell
        for x in range(grid_dims[0]):
             for y in range(grid_dims[1]):
                 if grid_counts[y, x] > 0:
                     grid_colors[y, x] /= grid_counts[y, x]

        return grid_colors.astype(np.uint8), grid_dims, min_bound, max_bound
    
    @numba.njit
    def compute_grid_colors(points, colors, step_size):
        # Manually compute the min and max for each axis
        min_x, min_y, min_z = np.inf, np.inf, np.inf
        max_x, max_y, max_z = -np.inf, -np.inf, -np.inf
        for point in points:
            min_x = min(min_x, point[0])
            min_y = min(min_y, point[1])
            min_z = min(min_z, point[2])
            max_x = max(max_x, point[0])
            max_y = max(max_y, point[1])
            max_z = max(max_z, point[2])
        
        min_bound = np.array([min_x, min_y, min_z], dtype=np.float64)
        max_bound = np.array([max_x, max_y, max_z], dtype=np.float64)

        # Calculate grid dimensions
        grid_dims = np.ceil((max_bound - min_bound)[:2] / step_size).astype(np.int64)

        # Compute grid coordinates
        grid_coords = ((points - min_bound)[:,:2] / step_size).astype(np.int64)
        grid_coords = np.clip(grid_coords, 0, grid_dims - 1)

        # Pre-allocate the grid array
        grid_colors = np.zeros((grid_dims[1], grid_dims[0], 3), dtype=np.uint8)

        # Assign colors to the grid
        for i in range(len(points)):
            grid_x, grid_y = grid_coords[i]
            grid_colors[grid_y, grid_x] = (colors[i] * 255).astype(np.uint8)

        return grid_colors, grid_dims, min_bound, max_bound

    def rasterize_single_segment(self, segment, output_folder, segment_idx, step_size):
        """
        Rasterizes the provided segment point cloud into a 2D image and saves it in the specified output folder.

        Args:
            pcd (open3d.geometry.PointCloud): The input point cloud to be rasterized.
            step_size (float): The step size for the grid used in rasterization.
            output_folder (str): The folder where the rasterized image will be saved.
            cluster_idx (int): The index of the cluster, used for naming the output file.

        Returns:
            tuple: A tuple containing grid dimensions, minimum and maximum bounds, and step size.
        """
        # Ensure the point cloud is not empty
        if segment.is_empty():
            raise ValueError("The point cloud is empty. Please check the file path and contents.")

        # Convert the point cloud to numpy arrays for points and colors
        points = np.asarray(segment.points)
        colors = np.asarray(segment.colors)

        # # Sort points by increasing x and then by increasing y
        # idx_sorted = np.lexsort((points[:,1], points[:,0]))  # Sort by y, then by x
        # sorted_points = points[idx_sorted]
        # sorted_colors = colors[idx_sorted]

        # Obtain grid colors with sorted points and colors
        grid_colors, grid_dims, min_bound, max_bound = PointCloudProcessor.compute_grid_colors2(points, colors, step_size)

        # Convert the array of colors into an image
        image = Image.fromarray(grid_colors)

        # Construct the file name using the cluster index
        file_name_img = os.path.join(output_folder, f'segment_{segment_idx}.png')
        image.save(file_name_img)

        # Display the rasterized image in the Jupyter notebook output
        print(f"Segment {segment_idx}: ")
        display(image)

        return grid_dims, min_bound, max_bound
    
    def rasterize_segments(self, step_size=0.25):
        """
        Projects each cluster onto a 2D plane using PCA alignment and rasterization, saves the results,
        and records all segment data in a single file.

        Parameters:
        step_size (float): The step size for the grid used in rasterization.
        """

        # Parse the filename from the original filepath to create a unique directory for the output
        _, file_name = os.path.split(self.file_path)
        base_name, _ = os.path.splitext(file_name)

        # Define the base directory for storing images
        base_directory = Path('D:/flowvis_data/results')

        # Create an output folder specifically for this set of cluster images
        self.output_folder = base_directory / f"{base_name}_2D"
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Open a file to write all segment data
        with open(os.path.join(self.output_folder, "segments_data.txt"), 'w') as data_file:
            # Iterate through each segmented cluster
            for idx, segment in enumerate(self.segments):
                
                # Align the cluster using PCA and store PCA data
                rotated_segment = self.pca_pointcloud(segment, idx)
                # Check if PCA data was correctly store in the dictionary
                if 'mean' not in self.segment_data_2d[idx] or 'eigenvectors' not in self.segment_data_2d[idx]:
                    print(f"PCA data missing for segment index {idx}")
                    continue  # Skip this segment
                # Append rotated segment to the list (rotated segments will be later used for reprojection to 3D space)
                self.segments_pca.append(rotated_segment)

                # Rasterize the aligned segment and retrieve relevant data
                grid_dims, min_bound, max_bound = self.rasterize_single_segment(rotated_segment, self.output_folder, idx, step_size=step_size)

                # Update the dictionary with the rasterization data without overwriting existing data
                self.segment_data_2d[idx].update({
                    'grid_dims': grid_dims,
                    'min_bound': min_bound,
                    'max_bound': max_bound,
                    'step_size': step_size
                })

                # Write the segment data, including PCA and rasterization data, into the file
                data_file.write(f"Segment Index: {idx}\n")
                data_file.write("Segment Data:\n")
                data_file.write("---------------\n")
                data_file.write(f"Grid Dimensions: {grid_dims}\n")
                data_file.write(f"Minimum Bound: {min_bound}\n")
                data_file.write(f"Maximum Bound: {max_bound}\n")
                data_file.write(f"Step Size: {step_size}\n")
                data_file.write(f"PCA Mean: {self.segment_data_2d[idx]['mean']}\n")
                data_file.write(f"PCA Eigenvectors:\n {self.segment_data_2d[idx]['eigenvectors']}\n\n")

        # Notify the user that the process is complete and where the files are saved
        print(f"Segment projection images and data have been saved in: {self.output_folder}")

    ##################################################################################################################################################################################
    # 2D SEGMENT ANALYSIS 
    ##################################################################################################################################################################################

    def apply_adaptive_thresholding(self, grayscale_image):
        """
        Applies adaptive thresholding to a grayscale image to convert it to a binary image.
        
        Args:
        - grayscale_image: A numpy array representing a grayscale image.

        Returns:
        - binary_image: A binary image after applying adaptive thresholding.

        The method performs the following steps:
        1. Apply a median blur to the grayscale image to reduce noise. This helps in achieving
           a more accurate thresholding. The kernel size of 5 is chosen for the median blur.
        2. Apply adaptive thresholding to the blurred image. Adaptive thresholding adjusts the
           threshold value based on the local pixel neighborhood. 
        """
        # Step 1: Apply Median Blur
        # blurred_image = cv2.medianBlur(grayscale_image, 5)
        # Step 2: Apply Adaptive Thresholding
        binary_image = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 11, self.c)

        return binary_image
    
    def remove_borders(self, grayscale_image, binary_image, kernel_size=9, iterations=3):
        """
        This function removes both external and internal (holes) borders from a binary image. It first identifies the largest 
        contour in the grayscale image, assuming it to be the external border, and then applies erosion to shrink this border.
        Additionally, it identifies internal borders (smaller contours) and removes them. The final result is a binary image
        with both external and internal borders removed. The function also displays intermediate steps for visual verification.

        Parameters:
        grayscale_image (numpy.ndarray): The grayscale image used to find the borders.
        binary_image (numpy.ndarray): The binary image from which the borders will be removed.
        kernel_size (int): The size of the kernel used for the erosion process.
        iterations (int): The number of erosions applied to the external border mask.

        Returns:
        numpy.ndarray: The binary image with both external and internal borders removed.
        """
        # Find the external contours of the grayscale image. The external contours are assumed to be the borders.
        contours, _ = cv2.findContours(grayscale_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Find the largest contour by area, which is assumed to be the external border of the image.
        external_contour = max(contours, key=cv2.contourArea)

        # Step 1: remove external borders
        # Create a mask with the same dimensions as the grayscale image. Initially, the mask is filled with zeros (black).
        mask_ext = np.zeros_like(grayscale_image)
        # Draw the external contours on the mask with white color (255) and fill the area inside the contour.
        cv2.drawContours(mask_ext, [external_contour], -1, color=255, thickness=cv2.FILLED)

        # Create a kernel for erosion. The kernel is a square matrix of ones with dimensions specified by kernel_size.
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Erode the mask to shrink the external contour. The erosion process is applied 'iterations' times.
        mask_ext_eroded = cv2.erode(mask_ext, kernel, iterations=iterations)
        # Subtract the eroded mask from the original mask to isolate the border area.
        mask_ext = cv2.subtract(mask_ext, mask_ext_eroded)
        # Invert the eroded mask to get the final mask that excludes the borders, for further processing of the binary image.
        extmask_ext_negative = cv2.bitwise_not(mask_ext)

        # Step 2: remove internal borders
        # Create a mask with the same dimensions as the grayscale image. Initially, the mask is filled with zeros (black).
        mask_int = np.zeros_like(grayscale_image)
        
        # Draw all contours on the mask and the contour image, except the external contour.
        for contour in contours:
            if cv2.contourArea(contour) != cv2.contourArea(external_contour):
                cv2.drawContours(mask_int, [contour], -1, color=255, thickness=8)

        # Invert the mask to get the negative mask
        mask_int_negative = cv2.bitwise_not(mask_int)

        # Combine the two masks using bitwise OR
        combined_mask = cv2.bitwise_and(extmask_ext_negative, mask_int_negative)
        # Apply the combined mask
        result_img = cv2.bitwise_and(binary_image, binary_image, mask=combined_mask)

        return result_img
    
    @staticmethod
    def clahe_gamma(image, clip_limit=1.2, tile_grid_size=(8, 8), gamma=0.90):
        """
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and gamma correction 
        to an image to enhance its contrast.

        Parameters:
        filepath (str): File path of the image to be processed.
        clip_limit (float): Threshold for contrast limiting in CLAHE.
        tile_grid_size (tuple of int): Size of grid for histogram equalization.
        gamma (float): Gamma value for correction. Values < 1 will increase contrast.

        Returns:
        numpy.ndarray: The processed image with enhanced contrast.
        """

        # Convert the image from BGR to Lab color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        # Split the Lab image into channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_channel_clahe = clahe.apply(l_channel)
        # Merge the CLAHE enhanced L channel with the original a and b channels
        lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
        # Convert the Lab image back to BGR color space
        image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_Lab2BGR)

        # Apply gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image_clahe_gamma = cv2.LUT(image_clahe, table)

        return image_clahe_gamma
    
    def remove_artifacts(self, filepath, binary_image, kernel_size=9, iterations=2):
        """
        Filters out red color from an image and applies the result to a binary image.

        Parameters:
        filepath (str): File path of the image to be processed.
        binary_image (numpy.ndarray): A binary image to which the red color filter will be applied.
        kernel_size (int): Size of the kernel for dilation.
        iterations (int): Number of iterations for dilation.

        Returns:
        numpy.ndarray: The filtered image with red colors blacked-out.
        numpy.ndarray: The binary image with red colors blacked-out.
        """
        # Read the image from the given file path
        image_read = cv2.imread(filepath)
        # Apply CLAHE and gamma correction to enhance the image contrast
        image = PointCloudProcessor.clahe_gamma(image_read)

        # Convert the enhanced image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define the range of red color in HSV
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([168, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # Threshold the HSV image to get only red colors
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Check if the image has red color; if not, return the original images
        if cv2.countNonZero(red_mask) == 0:
            return binary_image

        # Create a structuring element (kernel) for dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Dilate the red color mask
        dilated_mask = cv2.dilate(red_mask, kernel, iterations=iterations)
        # Invert the dilated mask to filter out the red color
        dilated_mask_inv = cv2.bitwise_not(dilated_mask)

        # Black-out the areas with red in the binary image
        filtered_binary_image = cv2.bitwise_and(binary_image, binary_image, mask=dilated_mask_inv)

        return filtered_binary_image
    
    def remove_small_components(self, image):
        """
        This function filters out small connected components (blobs) in a binary image based on a size threshold.
        
        Args:
            image (numpy.ndarray): The binary image from which to remove small components.
            size_threshold (int): The size threshold for connected components. Components with a count of white pixels
                                less than this threshold will be removed.
        
        Returns:
            numpy.ndarray: The binary image with small components removed.
        """
        
        # Validate input image
        if image is None or image.dtype != np.uint8:
            raise ValueError("The input must be a binary image with uint8 type.")
        
        # Find all connected components in the image using the function cv2.connectedComponents 
        num_labels, labels_im = cv2.connectedComponents(image)
        # Measure the size (i.e., the number of pixels) of each component
        component_sizes = np.bincount(labels_im.flatten())
        
        # Create a mask with the same size as the input image, initialized to zeros (i.e., black).
        filtered_image = np.zeros_like(image)
        # Loop through each component label.
        for i in range(1, num_labels):  # Start from 1 to ignore the background
            # If the component size is greater than or equal to the size threshold, mark it in the mask.
            if component_sizes[i] >= self.size_threshold:
                filtered_image[labels_im == i] = 255
        
        # Return the filtered mask, which will have removed components smaller than the size threshold.
        return filtered_image
    
    def cluster_components(self, image, connectivity=4):
        """
        Find and display connected components in an image, each in a random color.

        Parameters:
        image (ndarray): The binary image to process, where non-zero pixels are considered foreground.
        connectivity (int): The integer defining the pixel connectivity startegy 

        Returns:
        tuple: A tuple containing:
            - num_labels (int): The number of unique connected components found in the image.
            - labels (ndarray): An array the same size as the input image, where each element
                                has a value that corresponds to the component label.
            - stats (ndarray): A matrix with stats about each label, including the bounding box
                            and area of the connected components.
            - centroids (ndarray): An array with the centroid position for each label.
        """

        # Find connected components with statistics.
        num_labels, labels, _ , _ = cv2.connectedComponentsWithStats(image, connectivity=connectivity)

        # Create an output image to draw the components in color.
        labeled_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # Assign a random color to each component, excluding the background.
        for label in range(1, num_labels):  # Start from 1 to skip the background label.
            mask = labels == label
            # Generate a random color, excluding black (0,0,0).
            random_color = np.random.randint(1, 256, size=3)
            # Apply the color to the component in the output image.
            labeled_img[mask] = random_color

        # Return the connected components data.
        return labeled_img
    
    def extract_features(self, filepath):
        """
        Provisional method for extracting features from an image. This implementation is a placeholder and is subject to ongoing updates.

        This method performs a series of image processing operations to extract features from a grayscale image. It involves adaptive 
        thresholding, border removal, filtering small components, and clustering connected components.

        Args:
        - filepath (str): The path to the image file.

        Returns:
        - labeled_img (numpy.ndarray): An image with different connected components labeled in unique colors.

        The method performs the following steps:
        1. Reads the image in grayscale.
        2. Applies adaptive thresholding to convert the grayscale image to a binary format.
        3. Removes the external borders from the binary image and prepares it for further analysis.
        4. Filters out small components based on a size threshold.
        5. Clusters connected components in the filtered image and labels each cluster with a unique color.
        """

        # Step 1: Read the image in grayscale
        grayscale_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        # Apply median blur to grayscale image
        grayscale_image = cv2.medianBlur(grayscale_image,5)
        # Step 2: Apply adaptive thresholding
        binary_image = self.apply_adaptive_thresholding(grayscale_image)
        # Step 3: Remove borders from the binary image
        nobrdr_image = self.remove_borders(grayscale_image, binary_image)
        # Step 4: Remove artifacts (stickers)
        clean_image = self.remove_artifacts(filepath, nobrdr_image)
        # Step 5: Filter out small components
        filtered_image = self.remove_small_components(clean_image)
        # Step 6: Cluster and label connected components
        labeled_image = self.cluster_components(filtered_image, connectivity=4)

        # Convert the NumPy array to a PIL Image object for displaying: ensure the array is in the correct data type (uint8) and scale if necessary
        if labeled_image.dtype != np.uint8:
            labeled_img_copy = (255 * labeled_image).astype(np.uint8)
        else:
            labeled_img_copy = labeled_image
        display_img = Image.fromarray(labeled_img_copy)
        # Display the image in the Jupyter notebook output
        display(display_img)

        # Return the image with labeled features.
        return labeled_image
    
    def process_image(self, filepath):
        """
        Container method to process an image and save it with a modified filename. If a processed image already exists,
        the file will be overwritten. 

        Args:
        - filepath (str): The file path of the image to be processed. The path must be a valid path to an image file.

        Returns:
        - str: The file path of the processed image. This path reflects the new filename with the '_processed' suffix.

        """

        # Placeholder for processing the image
        processed_image = self.extract_features(filepath)

        # Generate a dynamic filename for the processed image
        base, ext = os.path.splitext(filepath)
        new_filename = f"{base}_processed{ext}"

        # Save the processed image, overwriting existing '_processed' file if it exists
        cv2.imwrite(new_filename, processed_image)

        return new_filename

    ##################################################################################################################################################################################
    # 2D-3D INTERFACE
    ################################################################################################################################################################################## 

    def conversion2d3d_prealloc(self, image_path, segment_idx):
        """
        Converts a 2D rasterized image back to a 3D point cloud representation.
        This function reconstructs a 3D point cloud from a 2D rasterized image, utilizing efficient computational techniques such as pre-allocation of memory and vectorization of operations for improved performance. It leverages the original 3D points as a spatial reference for accurate reconstruction.

        Args:
            image_path (str): The file path of the 2D rasterized image.
            segment_idx (int): The index of the segment to be converted back to 3D.

        Process Overview:
        1. Load the rasterized image and identify non-background pixels, which represent the 2D projection of the 3D points.
        2. Use a KDTree for efficient nearest neighbor searches in the original 3D point cloud, focusing on x and y coordinates.
        3. For each 2D point, find the nearest neighbors in 3D space. Interpolate the z-coordinate by averaging the z-coordinates of these neighbors, utilizing vectorized operations for efficiency.
        4. Reconstruct the 3D point cloud with these new points, maintaining the original color information from the rasterized image. Memory pre-allocation is used to optimize the reconstruction process.

        Returns:
            open3d.geometry.PointCloud: The reconstructed 3D point cloud that closely approximates the original point cloud's geometry and color, achieved through efficient computational methods.
        """

        # Retrieve the relevant data for the specified segment
        segment_data = self.segment_data_2d[segment_idx]
        step_size, min_bound = segment_data['step_size'], segment_data['min_bound']
        segment = self.segments_pca[segment_idx]

        # Load the rasterized image and convert it to a numpy array
        rasterized_image = Image.open(image_path)
        rasterized_pixels = np.array(rasterized_image)

        # Load the original 3D points of the segment
        segment_points = np.asarray(segment.points)

        # KDTree for nearest neighbor search, using only the x and y coordinates
        kdtree = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(segment_points[:, :2])

        # Identify non-background pixels (assuming black as background)
        non_bg_mask = np.any(rasterized_pixels != [0, 0, 0], axis=2)
        non_bg_indices = np.argwhere(non_bg_mask)
        num_non_bg_pixels = len(non_bg_indices)

        # If no non-background pixels are found, print the message and return an empty point cloud
        if num_non_bg_pixels == 0:
            print(f"UserWarning:\nNo non-background pixels found for segment {segment_idx}. "
                "Returning an empty flowvis features point cloud. "
                "It is likely that the segment under analysis presents no flowvis features. If this is not the case, it is recommended to reduce the current step-size when performing point cloud rasterization.\n")
            empty_pcd = o3d.geometry.PointCloud()
            return empty_pcd

        # Pre-allocate arrays for batch processing
        batch_points = np.zeros((num_non_bg_pixels, 2))
        batch_colors = np.zeros((num_non_bg_pixels, 3))

        # Populate the batch arrays with the corresponding 2D coordinates and colors
        for i, (y, x) in enumerate(non_bg_indices):
            grid_x = x * step_size + min_bound[0]
            grid_y = y * step_size + min_bound[1]  
            batch_points[i] = [grid_x, grid_y]
            batch_colors[i] = rasterized_pixels[y, x] / 255.0

        # Query the KDTree with the batch of 2D points to find nearest neighbors in 3D
        _ , indices = kdtree.kneighbors(batch_points)

        # Calculate the z-coordinate by averaging the z-coordinates of nearest neighbors
        z_coords = segment_points[indices, 2]
        z_interp = np.mean(z_coords, axis=1)
        new_point_cloud_data = np.hstack((batch_points, z_interp[:, np.newaxis]))

        # Create a new point cloud object with the reconstructed 3D points and colors
        features_pcd = o3d.geometry.PointCloud()
        features_pcd.points = o3d.utility.Vector3dVector(new_point_cloud_data)
        features_pcd.colors = o3d.utility.Vector3dVector(batch_colors)

        return features_pcd
    
    def process_segments(self, c=-3, size_threshold=50):
        """
        Processes each 2D segment image in the specified output folder and maps the processed 2D data
        back to the original 3D space.
        This function ensures that the order of segments in the 3D space is maintained according to their indices.

        Args:
            c (int): A parameter used in other processing methods (context-specific).
            size_threshold (int): Another parameter for processing, context-specific.

        The function performs the following operations:
        1. Iterates over segment images in the output folder, processing only those that meet the naming criteria and skipping already processed images.
        2. Sorts these images based on their segment index to maintain the order.
        3. Processes each image, maps it back to 3D space, and stores the results in an ordered manner.
        """

        # Assign the given parameters to the class attributes for use in other methods
        self.c = c
        self.size_threshold = size_threshold

        # Prepare a list to hold tuples of (segment index, filename)
        segment_files = []
        for filename in os.listdir(self.output_folder):
            # Skip already processed files and files that don't match the segment naming pattern
            if "_processed" in filename or not (filename.endswith(".png") and filename.startswith("segment_")):
                continue

            # Extract the segment index from the filename
            segment_idx = int(filename.split('_')[1].split('.')[0])

            # Append the tuple of segment index and filename to the list
            segment_files.append((segment_idx, filename))

        # Sort the list of tuples based on the segment index to ensure correct order
        segment_files.sort(key=lambda x: x[0])

        # Process each file in the sorted list
        for segment_idx, filename in segment_files:
            # Construct the full file path
            filepath = os.path.join(self.output_folder, filename)

            # Notify the user which segment is being processed
            print(f"Segment {segment_idx}: ")

            # Process the image and get the path of the processed image
            processed_img_filepath = self.process_image(filepath)

            # Convert the processed image back to 3D space(in PCA reference system)
            features_pcd_pca = self.conversion2d3d_prealloc(processed_img_filepath, segment_idx)
            # revert back from PCA to original reference system
            features_pcd_original = self.revert_pca_transformation(features_pcd_pca, segment_idx)

            # Append the processed 3D features to the class attribute list
            self.segment_features.append(features_pcd_original)

        # Inform the user of the completion of the process and the location of the saved files
        print(f"Processed images have been saved in: {self.output_folder}")

    @staticmethod
    @jit(nopython=True)
    def upsample_points_and_colors_numba(original_points, original_colors, num_points_per_original_point, radius):
        """
        Numba-accelerated function to upsample a point cloud and copy the color of original points to new points. 

        Args:
        - original_points: Numpy array of original points.
        - original_colors: Numpy array of colors corresponding to original points.
        - num_points_per_original_point: Number of points to add per original point.
        - radius: Radius within which new points are added around each original point.

        Returns:
        - upsampled_points: Numpy array of upsampled points.
        - upsampled_colors: Numpy array of colors for the upsampled points.
        """
        num_original_points = original_points.shape[0]
        total_points = num_original_points * (num_points_per_original_point + 1)
        upsampled_points = np.empty((total_points, 3))
        upsampled_colors = np.empty((total_points, 3))

        for i in range(num_original_points):
            base_index = i * (num_points_per_original_point + 1)
            upsampled_points[base_index] = original_points[i]
            upsampled_colors[base_index] = original_colors[i]
            for j in range(1, num_points_per_original_point + 1):
                random_offset = radius * np.random.randn(3)
                upsampled_points[base_index + j] = original_points[i] + random_offset
                upsampled_colors[base_index + j] = original_colors[i]

        return upsampled_points, upsampled_colors

    def upsample_pcd(self, pcd, points_upsample, radius=0.01):
        """
        Upsamples a given point cloud by adding multiple points around each original point. This function is intended 
        to highlight the features point cloud, therefore for mere display purposes. 

        Args:
        - pcd: The original point cloud to be upsampled.
        - num_points_per_original_point: Number of new points to generate per original point in the cloud.
        - radius: The radius within which new points are randomly distributed around each original point.

        Returns:
        - upsampled_pcd: The upsampled point cloud.
        """

        # Convert point cloud data to numpy arrays for processing.
        original_points = np.asarray(pcd.points)
        # Check if the point cloud has color information
        original_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(original_points)

        # Call the static method upsample_points_and_colors_numba to create the upsampled points and their colors.
        upsampled_points, upsampled_colors = PointCloudProcessor.upsample_points_and_colors_numba(
            original_points, original_colors, points_upsample, radius
        )

        # Create a new point cloud object for the upsampled points.
        upsampled_pcd = o3d.geometry.PointCloud()
        # Assign the upsampled points and their colors to the new point cloud.
        upsampled_pcd.points = o3d.utility.Vector3dVector(upsampled_points)
        upsampled_pcd.colors = o3d.utility.Vector3dVector(upsampled_colors)

        # Return the upsampled point cloud.
        return upsampled_pcd


    def results_3d(self, points_upsample=50, full_res=True):
        """
        This function performs display and storing operations for the final 3D point cloud:
        1. Displays each segment of the point cloud with its corresponding features, upsampled for display purposes.
        2. Combines all upsampled and non-upsampled segment features into a single point cloud for each
        3. Visualizes the original point cloud with all features combined (upsampled for display purposes).
        4. Saves the combined non-upsampled features point cloud as a .ply file in the specified output folder.

        It utilizes PointCloudVisualizer for visualizing point clouds at different stages of the process.

        Args:
        - points_upsample (int): The number of points around each point from the features point cloud for the upsampling operation (display purposes)

        """
        # Initialize an empty point cloud for combined features, both original and upsampled
        combined_features_pcd = o3d.geometry.PointCloud()
        combined_features_pcd_upsampled = o3d.geometry.PointCloud()

        # Loop through each segment and its features with index
        for idx in range(len(self.segments)):
            # retrieve corresponding segment and features
            segment = self.segments[idx]
            features = self.segment_features[idx]
            # Print the current segment index
            print(f"Segment {idx}: ")
            # Upsample features point cloud to highlight it (mere display purposes)
            features_upsampled = self.upsample_pcd(features, points_upsample)
            # Visualize the segment and its corresponding features
            PointCloudVisualizer([features_upsampled, segment])
            # Combine points and colors from each original features point cloud
            combined_features_pcd += features
            # Combine points and colors from each upsampled features point cloud
            combined_features_pcd_upsampled += features_upsampled

        # Display the original point cloud along with all features combined
        print("Complete Point Cloud: ")
        # use the downsampled point cloud in the final visualization for better handling and to avoid memory errors
        if full_res:
            PointCloudVisualizer([self.full_res_pcd_copy, combined_features_pcd_upsampled])
        else:
            PointCloudVisualizer([self.input_cloud_copy, combined_features_pcd_upsampled])

        # Store the resulting feature pcd (not upsampled) in the output folder
        # Extract the original file name without extension
        original_filename = os.path.splitext(os.path.basename(self.file_path))[0]
        # Construct the new file name with '_features.ply'
        new_file_name = f"{original_filename}_features.ply"
        # Full path for the new file
        output_file_path = os.path.join(self.output_folder, new_file_name)
        # Save the features point cloud (not upsampled)
        o3d.io.write_point_cloud(output_file_path, combined_features_pcd)

        # Print out the path of the saved file (optional)
        print(f"Combined features point cloud saved as: {output_file_path}")

    ##################################################################################################################################################################################
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # FLOW-VIS DETECTION AND CLASSIFICATION - 3D APPROACH (DISCARDED)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ##################################################################################################################################################################################

    ##################################################################################################################################################################################
    # ADAPTIVE METHODS
    ################################################################################################################################################################################## 

    @staticmethod
    def process_voxel(args):
        """
        Process a single voxel using adaptive thresholding, i.e. calculate the voxel mean, set the threshold by subtracting an arbitrary, user-defined, constant c to the mean and threhsold the voxels' colors.

        Args:
            args (tuple): Tuple containing voxel information and processing parameters.

        Returns:
            tuple: A tuple containing mask and tile after adaptive thresholding.
        """
        voxel_grid, voxel, c, colors, xyz = args

        # Obtain voxel bounding points
        voxel_bounding_points = voxel_grid.get_voxel_bounding_points(voxel.grid_index)
        min_bound = np.min(np.array(voxel_bounding_points), axis=0)
        max_bound = np.max(np.array(voxel_bounding_points), axis=0)
        # Extract points within the voxel
        mask = (
            (xyz[:, 0] >= min_bound[0]) & (xyz[:, 0] < max_bound[0]) &
            (xyz[:, 1] >= min_bound[1]) & (xyz[:, 1] < max_bound[1]) &
            (xyz[:, 2] >= min_bound[2]) & (xyz[:, 2] < max_bound[2])
        )
        # Extract colors within the voxel
        tile = colors[mask]
        tile = np.round(tile).astype(int)  # convert to integer
        # Calculate threshold based on the tile's mean and clip_limit
        threshold = np.mean(tile) - c
        # Apply adaptive thresholding
        tile = np.where(tile < threshold, 0, 255)

        return mask, tile

    def adaptiveThreshold_3d_parallel(self, c, voxel_size, num_cores):
        """
        Perform parallel adaptive thresholding on a 3D point cloud. Create a voxel grid onto the input point cloud and perform adaptive thresholding on each voxel in parallel. 

        Args:
            c (float): Adaptive threshold parameter.
            voxel_size (float): Voxel size for creating a voxel grid.
            num_cores (int): Number of CPU cores to use for parallel processing.

        Returns:
            o3d.geometry.PointCloud: Point cloud with highlighted "white" points.
        """
        xyz = np.asarray(self.input_cloud.points)
        colors = np.asarray(self.input_cloud.colors) * 255
        thresholded_colors = np.zeros_like(colors)

        # Create a Voxel Grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.input_cloud, voxel_size=voxel_size)

        # Initialize thresholded colors variable
        thresholded_colors = np.zeros_like(colors)

        # Create a ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            tasks = []

            # Process each voxel in parallel
            for voxel in voxel_grid.get_voxels():
                args = (voxel_grid, voxel, c, colors, xyz)
                task = executor.submit(self.process_voxel, args)
                tasks.append(task)

            # Retrieve results from completed tasks
            for task in tasks:
                result = task.result()
                if result is not None:
                    mask, tile = result
                    if mask is not None:
                        thresholded_colors[mask] = tile

        # Convert colors to open3d format
        thresholded_colors = thresholded_colors / 255.0
        # Create a mask for points that are below the threshold (black points)
        below_threshold_mask = (thresholded_colors[:, 0] == 0.0) & (thresholded_colors[:, 1] == 0.0) & (thresholded_colors[:, 2] == 0.0)
        # Create a new point cloud for "white" points
        white_points_cloud = self.input_cloud.select_by_index(np.where(~below_threshold_mask)[0])
        # Visualize the result with white points in red and "black" points with original colors
        # white_points_cloud_copy = copy.deepcopy(white_points_cloud)
        # white_points_cloud_copy.paint_uniform_color([1.0, 0.0, 0.0])
        #o3d.visualization.draw_geometries([white_points_cloud, self.input_cloud_copy])
    
        # Return "white" point clouds
        return white_points_cloud

    def adaptiveThreshold_clusters(self, c):
        """
        Apply adaptive thresholding to the input cluster based on colors. Adaptive thresholding is performed on the whole cluster and voxelization is not carried out. 

        Args:
            c (float): Constant value for threshold calculation.

        Returns:
            o3d.geometry.PointCloud: Point cloud containing only "white" points.
        """
        # Check if a point cloud is loaded
        if self.input_cloud is None:
            raise ValueError("No point cloud available. Load a point cloud first.")

        # Extract the (x, y, z) coordinates and colors from the input cluster
        colors = np.asarray(self.input_cloud.colors) * 255
        colors = np.round(colors).astype(int)  # Convert to integer
        # Calculate the threshold based on the cluster's mean and constant
        threshold = np.mean(colors) - c
        # Apply adaptive thresholding to the entire cluster
        colors = np.where(colors < threshold, 0, 255)
        # Normalize the thresholded values to be in the [0, 1] range
        colors = colors / 255.0
        # Create a mask for points that are below the threshold (black points)
        below_threshold_mask = (colors[:, 0] == 0.0) & (colors[:, 1] == 0.0) & (colors[:, 2] == 0.0)
        # Create a mask for points that are above the threshold (white points)
        above_threshold_mask = ~below_threshold_mask
        # Create a new point cloud for "white" points
        white_points = self.input_cloud.select_by_index(np.where(above_threshold_mask)[0])
        # Return the point cloud containing only "white" points
        return white_points
    
    ##################################################################################################################################################################################
    # CONTROL METHODS FOR FLOWVIS EXTRACTION
    ################################################################################################################################################################################## 

    @staticmethod
    def process_cluster(cluster_args):
        """
        Process a single cluster by applying adaptive thresholding in parallel.

        Args:
            cluster_args (tuple): Tuple containing information about the cluster and parameters.
                cluster (o3d.geometry.PointCloud): Input point cloud cluster.
                c (float): Constant value for threshold calculation.
                voxel_size (float): Voxel size for 3D adaptive thresholding (if applicable).
                voxel (bool): Flag indicating whether to use voxelization in adaptive thresholding.
                num_cores (int): Number of CPU cores to use in parallel processing.

        Returns:
            o3d.geometry.PointCloud: Point cloud containing only "white" points.
        """
        # Extract cluster and parameters from the input tuple
        cluster, c, voxel_size, voxel, num_cores = cluster_args
        # Create a PointCloudProcessor instance for the current cluster
        cluster_processor = PointCloudProcessor(None, cluster, cluster)
        # Check if 3D adaptive thresholding with voxelization is requested
        if voxel == False:
            # Apply 2D adaptive thresholding to the cluster
            white_points = cluster_processor.adaptiveThreshold_clusters(c)
        else:
            # Apply 3D adaptive thresholding to the cluster using parallel processing
            white_points = cluster_processor.adaptiveThreshold_3d_parallel(c, voxel_size, num_cores)
        # Return the resulting point cloud containing only "white" points
        return white_points


    def extract_flowvis_lines(self, voxel=True):
        """
        Extract flowvis lines from point cloud. Process clusters in parallel by applying adaptive thresholding, performed in parallel. 
        Method: adaptive thresholding on voxels created on clusters based on the point cloud normals:
        1st parallel step: process each cluster in parallel
        2nd parallel step (whithin the first): for each cluster, process each voxel in parallel (each voxel requires the same independent operation)

        Args:
            num_clusters (int): Number of clusters for point cloud normals-based clustering.
            c (float, optional): Constant value for threshold calculation. Default is -7.
            voxel_size (float, optional): Voxel size for 3D adaptive thresholding. Default is 0.5.
            voxel (bool, optional): Flag indicating whether to use voxelization in adaptive thresholding. Default is True.

        Raises:
            ValueError: If no point cloud is available, it raises an error.

        Returns:
            None: The method visualizes the result and assigns "white" points to the processed cloud.
        """
        if self.input_cloud is None:
            raise ValueError("No point cloud available. Load a point cloud first.")
        
        # Step 0: Calculate or set all required parameters:
        # the values set below are heuristic/empirical and were determined by a trial and error strategy: they represent "what works best"
        # a. calculate available cores and use the maximum number of cores for both parallel processes
        num_cores = multiprocessing.cpu_count()  
        # b. set the number of clusters based on normals
        num_clusters = 4  #detect upper and surfaces and diffeentiate between sides for side surfaces
        # c. set c constant value (adaptive thresholding) based on flowvis used
        # the c constant value establishes how aggressive/conservative the adaptive thresholding is: white flowvis requires a more stringent constant, while green flowvis requires a more relaxed one, due to luminosity matters
        if self.flowvis_color == "white":
            c = -7
        else:
            c = -3
        # d. set voxel size based on input point cloud size
        metrics = calculate_point_cloud_metrics(self.input_cloud)
    
        if any(dim > 2500.0 for dim in metrics["Bounding Box Dimensions (X, Y, Z)"]):
            voxel_size = 12.0
        elif any(dim > 2000.0 for dim in metrics["Bounding Box Dimensions (X, Y, Z)"]):
            voxel_size = 10.0
        elif any(dim > 1500.0 for dim in metrics["Bounding Box Dimensions (X, Y, Z)"]):
            voxel_size = 8.0
        else:
            voxel_size = 6.0

        # inform the end-user of the parameters' choice
        print(f"Number of CPU cores used: {num_cores}")
        print(f"Number of Clusters: {num_clusters}")
        print(f"Constant Value (c) for Adaptive Thresholding: {c}")
        print(f"Voxel Size: {voxel_size}")

        # Step 1: Cluster by point cloud normals
        clustered_point_clouds, _ = self.kmeans_normals_clustering(num_clusters)

        # Step 2: Prepare arguments for parallel processing
        cluster_args_list = [(cluster, c, voxel_size, voxel, num_cores) for cluster in clustered_point_clouds]

        # Step 3: Create a ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Step 3: Process clusters in parallel
            all_white_points = list(executor.map(self.process_cluster, cluster_args_list))

        # Step 4: Highlight white points in the original point cloud
        white_points = []
        white_points_colors = []
        white_points_normals = []
        for cluster_white in all_white_points:
            white_points.extend(cluster_white.points)
            white_points_colors.extend(cluster_white.colors)
            white_points_normals.extend(cluster_white.normals)

        # Create a new point cloud with white points painted red for highlight purposes
        white_points_cloud = o3d.geometry.PointCloud()
        white_points_cloud.points = o3d.utility.Vector3dVector(white_points)
        white_points_cloud.colors = o3d.utility.Vector3dVector(white_points_colors)
        white_points_cloud.normals = o3d.utility.Vector3dVector(white_points_normals)

        # Visualize the result with white points in red and "black" points with original colors
        white_points_cloud_copy = copy.deepcopy(white_points_cloud)
        white_points_cloud_copy.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.visualization.draw_geometries([white_points_cloud_copy, self.input_cloud_copy])

        # Assign white points to processed cloud for further processing
        self.input_cloud = white_points_cloud

    ##################################################################################################################################################################################
    # FLOW-VIS FEATURES IDENTIFICATION: DBSCAN
    ##################################################################################################################################################################################   

    def apply_dbscan_clustering(self, start_index=5000, threshold=0.00005, min_samples=3):
        """
        Apply DBSCAN clustering to the flowvis cloud and visualize the results.
        This method first finds an appropriate epsilon value using a K-distance graph and then applies DBSCAN clustering.

        Args:
            flowvis_cloud (o3d.geometry.PointCloud): The point cloud on which DBSCAN is to be applied.
            start_index (int): The starting index for analysis in the K-distance graph to exclude the initial steep part.
            threshold (float): The threshold for detecting the knee point in the K-distance graph.
            min_samples (int): The minimum number of samples in a neighborhood for a point to be considered a core point.

        Returns:
            o3d.geometry.PointCloud: The DBSCAN clustered point cloud with noise removed.
        """
        # Determine optimal epsilon value for DBSCAN operation
        eps = self.calculate_optimal_epsilon(self.input_cloud.points, start_index=start_index, threshold=threshold, n_neighbors=8, plot=True)

        # Step 4: DBSCAN Clustering
        if eps is not None:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            self.labels = dbscan.fit_predict(np.asarray(self.input_cloud.points))

            # Step 5: Remove Noise and Create Clustered Point Cloud
            valid_clusters = self.labels != -1
            filtered_points = np.asarray(self.input_cloud.points)[valid_clusters]
            filtered_normals = np.asarray(self.input_cloud.normals)[valid_clusters]
            filtered_labels = self.labels[valid_clusters]
            # update labels (running variable)
            self.labels = filtered_labels

            unique_labels = np.unique(filtered_labels)
            random_colors = np.random.rand(len(unique_labels), 3)
            label_to_color = dict(zip(unique_labels, random_colors))
            cluster_colors = [label_to_color[label] for label in filtered_labels]

            # update point cloud (running variable)
            self.input_cloud.points = o3d.utility.Vector3dVector(filtered_points)
            self.input_cloud.colors = o3d.utility.Vector3dVector(cluster_colors)
            self.input_cloud.normals = o3d.utility.Vector3dVector(filtered_normals)

            # Step 6: Visualize the Clustered Cloud
            o3d.visualization.draw_geometries([self.input_cloud, self.input_cloud_copy])
        else:
            print("No knee point detected with the given threshold.")
            return None
        
    def apply_spectral_clustering(self, n_clusters=4):
        # Assuming self.labels and self.input_cloud are already set by previous methods
        unique_labels = np.unique(self.labels)
        num_unique_labels = len(unique_labels)
        
        if num_unique_labels < n_clusters:
            print(f"Number of unique clusters ({num_unique_labels}) is less than the number of spectral clusters ({n_clusters}).")
            return

        cluster_data = {}
        for label in unique_labels:
            indices = np.where(self.labels == label)[0]
            points = np.asarray(self.input_cloud.points)[indices]
            
            # Calculate centroid and covariance
            centroid = np.mean(points, axis=0)
            covariance = np.cov(points.T)
            
            # Create a temporary point cloud to calculate the oriented bounding box
            cluster_cloud = o3d.geometry.PointCloud()
            cluster_cloud.points = o3d.utility.Vector3dVector(points)
            
            # Calculate the minimal oriented bounding box
            bbox = cluster_cloud.get_oriented_bounding_box()
            dimensions = bbox.extent
            sorted_dimensions = np.sort(dimensions)
            aspect_ratio = sorted_dimensions[2] / sorted_dimensions[1]  # Assuming sorted_dimensions[2] is the largest dimension

            cluster_data[label] = {'centroid': centroid, 'covariance': covariance, 'aspect_ratio': aspect_ratio}

        # Create the similarity matrix
        similarity_matrix = np.zeros((num_unique_labels, num_unique_labels))

        for i, label_i in enumerate(unique_labels):
            for j, label_j in enumerate(unique_labels):
                if i == j:
                    continue
                
                # Calculate distance based on covariance
                cov_dist = np.linalg.norm(cluster_data[label_i]['covariance'] - cluster_data[label_j]['covariance'])
                
                # Calculate distance based on aspect ratio difference
                aspect_ratio_dist = abs(cluster_data[label_i]['aspect_ratio'] - cluster_data[label_j]['aspect_ratio'])

                # Combine covariance distance and aspect ratio distance
                total_distance = cov_dist + aspect_ratio_dist
                
                # Update similarity matrix
                similarity_matrix[i, j] = 1 / (total_distance + 1e-5)

        spectral_clust = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        group_labels = spectral_clust.fit_predict(similarity_matrix)

        # Map spectral cluster labels to original cluster labels
        label_to_spectral = {label: group_labels[i] for i, label in enumerate(unique_labels)}

        # Create a mapping of points to spectral cluster labels
        spectral_labels = np.array([label_to_spectral.get(label, -1) for label in self.labels])

        # Update the point cloud colors based on the new spectral labels
        spectral_colors = np.random.rand(n_clusters, 3)
        point_colors = np.array([spectral_colors[spectral_label] if spectral_label != -1 else [0, 0, 0] for spectral_label in spectral_labels])

        self.input_cloud.colors = o3d.utility.Vector3dVector(point_colors)
        o3d.visualization.draw_geometries([self.input_cloud])

        return spectral_labels


    def filter_clusters_by_size(self, min_size=5):
        """
        Filter clusters in self.input_cloud based on their size (number of points).

        Args:
            min_size (int): Minimum number of points required for a cluster to be considered valid.

        Returns:
            None: Updates self.input_cloud with the filtered clusters.
        """
        filtered_points = []
        filtered_normals = []
        filtered_labels = []

        # obtain unique labels
        unique_labels = np.unique(self.labels)

        for label in unique_labels:
            cluster_mask = self.labels == label  # Use self.labels for mask
            cluster_points = np.asarray(self.input_cloud.points)[cluster_mask]
            cluster_normals = np.asarray(self.input_cloud.normals)[cluster_mask]

            if len(cluster_points) >= min_size:
                filtered_points.extend(cluster_points)
                filtered_normals.extend(cluster_normals)
                filtered_labels.extend([label] * len(cluster_points))

        # update labels (running variable)
        self.labels = filtered_labels

        unique_labels = np.unique(filtered_labels)
        random_colors = np.random.rand(len(unique_labels), 3)
        label_to_color = dict(zip(unique_labels, random_colors))
        cluster_colors = [label_to_color[label] for label in filtered_labels]

        # update point cloud (running variable)
        self.input_cloud.points = o3d.utility.Vector3dVector(filtered_points)
        self.input_cloud.colors = o3d.utility.Vector3dVector(cluster_colors)
        self.input_cloud.normals = o3d.utility.Vector3dVector(filtered_normals)

        # Step 6: Visualize the Clustered Cloud
        o3d.visualization.draw_geometries([self.input_cloud, self.input_cloud_copy])

    def filter_clusters_by_aspect_ratio(self, aspect_ratio_threshold=3.0):
        """
        Filter clusters in self.input_cloud based on their aspect ratio.

        Args:
            aspect_ratio_threshold (float): Minimum aspect ratio required for a cluster to be considered valid.

        Returns:
            None: Updates self.input_cloud with the filtered clusters.
        """
        filtered_points = []
        filtered_normals = []
        filtered_labels = []

        # obtain unique labels
        unique_labels = np.unique(self.labels)

        for label in unique_labels:
            cluster_mask = self.labels == label
            cluster_points = np.asarray(self.input_cloud.points)[cluster_mask]
            cluster_normals = np.asarray(self.input_cloud.normals)[cluster_mask]

            # Create a temporary point cloud for the current cluster
            cluster_cloud = o3d.geometry.PointCloud()
            cluster_cloud.points = o3d.utility.Vector3dVector(cluster_points)
            cluster_cloud.normals = o3d.utility.Vector3dVector(cluster_normals)

            # Calculate the minimal oriented bounding box
            try:
                bbox = o3d.geometry.OrientedBoundingBox.create_from_points(cluster_cloud.points, robust=True)
            except RuntimeError as e:
                print(f"Skipping cluster {label} due to error: {e}")
                continue

            # Get the dimensions of the bounding box and calculate aspect ratio
            dimensions = bbox.extent
            sorted_dimensions = np.sort(dimensions)
            aspect_ratio = sorted_dimensions[2] / sorted_dimensions[1]

            # Skip clusters below the aspect ratio threshold
            if aspect_ratio < aspect_ratio_threshold:
                continue

            # Add the cluster to the filtered list if it meets the aspect ratio threshold
            filtered_points.extend(cluster_points)
            filtered_normals.extend(cluster_normals)
            filtered_labels.extend([label] * len(cluster_points))

        # update labels (running variable)
        self.labels = filtered_labels

        unique_labels = np.unique(filtered_labels)
        random_colors = np.random.rand(len(unique_labels), 3)
        label_to_color = dict(zip(unique_labels, random_colors))
        cluster_colors = [label_to_color[label] for label in filtered_labels]

        # update point cloud (running variable)
        self.input_cloud.points = o3d.utility.Vector3dVector(filtered_points)
        self.input_cloud.colors = o3d.utility.Vector3dVector(cluster_colors)
        self.input_cloud.normals = o3d.utility.Vector3dVector(filtered_normals)

        # Step 6: Visualize the Clustered Cloud
        o3d.visualization.draw_geometries([self.input_cloud, self.input_cloud_copy])

    ##################################################################################################################################################################################
    # FLOW-VIS FEATURES ABSTRACTION
    ##################################################################################################################################################################################   

    def fit_polynomial_curves_to_clusters(self):
        """
        Fit polynomial curves to each cluster in self.input_cloud.

        Returns:
            None: Updates self.input_cloud with the fitted polynomial curves.
        """
        # Lists to store lines and cylinders for each cluster
        all_lines = []
        all_cylinders = []

        # obtain unique labels
        unique_labels = np.unique(self.labels)

        for label in unique_labels:
            cluster_mask = self.labels == label
            cluster_points = np.asarray(self.input_cloud.points)[cluster_mask]

            # Skip processing if the cluster is too small
            if len(cluster_points) < 5:
                continue

            # Extract lines and cylinders from the current cluster
            line_set, cylinders = self.extract_curve(cluster_points)

            # Store lines and cylinders if available
            if line_set is not None:
                all_lines.append(line_set)
                all_cylinders.extend(cylinders)

        # Merge all LineSets into a single LineSet
        all_lines_set = o3d.geometry.LineSet()
        for line_set in all_lines:
            all_lines_set += line_set

        # Merge all TriangleMeshes into a single TriangleMesh
        all_cylinders_mesh = o3d.geometry.TriangleMesh()
        for cylinder in all_cylinders:
            all_cylinders_mesh += cylinder

        # Visualize all LineSets and TriangleMeshes together
        geometries_to_visualize = [all_lines_set, all_cylinders_mesh, self.input_cloud, self.input_cloud_copy]
        o3d.visualization.draw_geometries(geometries_to_visualize)

    @staticmethod
    def fit_curve_xz(points):
        """
        Fit a cubic polynomial to the x-z plane of the given points.

        Args:
            points (np.ndarray): Numpy array of points with shape (N, 3).

        Returns:
            tuple: Contains sorted x-values (t), original z-values (z), fitted z-values (z_fit),
                   and polynomial coefficients (coeffs_xz).
        """
        # Extract x and z coordinates
        t = points[:, 0]
        z = points[:, 2]

        # Sort points by x coordinate for a meaningful polynomial fit
        sorted_indices = np.argsort(t)
        t = t[sorted_indices]
        z = z[sorted_indices]

        # Fit a 4th degree cubic polynomial and evaluate it
        coeffs_xz = np.polyfit(t, z, 4)
        z_fit = np.polyval(coeffs_xz, t)

        return t, z, z_fit, coeffs_xz
    
    @staticmethod
    def fit_curve_xy(points):
        """
        Fit a cubic polynomial to the x-y plane of the given points.

        Args:
            points (np.ndarray): Numpy array of points with shape (N, 3).

        Returns:
            tuple: Contains sorted x-values (t), original y-values (y), fitted y-values (y_fit),
                   and polynomial coefficients (coeffs_xy).
        """
        # Extract x and y coordinates
        t = points[:, 0]
        y = points[:, 1]

        # Sort points by x coordinate for a meaningful polynomial fit
        sorted_indices = np.argsort(t)
        t = t[sorted_indices]
        y = y[sorted_indices]

        # Fit a 3rd degree cubic polynomial and evaluate it
        coeffs_xy = np.polyfit(t, y, 3)
        y_fit = np.polyval(coeffs_xy, t)

        return t, y, y_fit, coeffs_xy
    
    @staticmethod
    def combine_curves(x_fit, y_fit, z_fit):
        """
        Combine x, y, and z coordinates to form a 3D curve.

        Args:
            x_fit (np.ndarray): Fitted x-values.
            y_fit (np.ndarray): Fitted y-values.
            z_fit (np.ndarray): Fitted z-values.

        Returns:
            np.ndarray: Combined 3D curve with shape (N, 3).
        """
        # Combine the fitted coordinates into a single array representing a 3D curve
        return np.column_stack((x_fit, y_fit, z_fit))
    
    @staticmethod
    def create_cylinder(start_point, direction):
        """
        Create a cylinder geometry aligned along a specified direction starting from a given point.

        Args:
            start_point (np.ndarray): The start point of the cylinder.
            direction (np.ndarray): The direction vector of the cylinder.

        Returns:
            o3d.geometry.TriangleMesh: The created cylinder geometry.
        """
        length = np.linalg.norm(direction)

        # Return None if the direction vector has zero length (avoids division by zero)
        if length <= 0:
            return None

        # Create a cylinder with specified dimensions
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.2, height=length*2)
        cylinder.translate(start_point)

        # Calculate rotation to align the cylinder with the direction vector
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_axis, direction) / length)
        rotation_matrix = Rotation.from_rotvec(axis * angle).as_matrix()

        # Rotate the cylinder to align with the direction vector
        cylinder.rotate(rotation_matrix, center=start_point)

        return cylinder
    
    def extract_curve(self, cluster_points):
        """
        Extract a curve from a cluster of points using polynomial fitting.

        Args:
            cluster_points (np.ndarray): Points of the cluster.

        Returns:
            tuple: A line set representing the curve and a list of cylinder geometries.
        """
        # Perform Principal Component Analysis (PCA) on the cluster
        mean_point = np.mean(cluster_points, axis=0)
        centered_points = cluster_points - mean_point
        covariance_matrix = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Order eigenvalues and eigenvectors by importance (descending eigenvalues)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Rotate points into PCA coordinate system
        rotated_points = np.dot(centered_points, eigenvectors)

        # Fit cubic polynomials in the PCA coordinate system
        t_xz, _ , z_fit, _ = self.fit_curve_xz(rotated_points)
        _ , _ , y_fit, _ = self.fit_curve_xy(rotated_points)

        # Combine the fitted curves to form a 3D curve
        curve_3d = self.combine_curves(t_xz, y_fit, z_fit)

        # Rotate the 3D curve back to the original coordinate system
        rotated_curve_3d = np.dot(curve_3d, eigenvectors.T) + mean_point

        # Create a point cloud and a line set from the 3D curve
        curve_3d_point_cloud = o3d.geometry.PointCloud()
        curve_3d_point_cloud.points = o3d.utility.Vector3dVector(rotated_curve_3d)

        # Create a line set from the curve points
        lines = [[i, i + 1] for i in range(len(rotated_curve_3d) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = curve_3d_point_cloud.points
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1, 0, 0]]), (len(lines), 1)))

        # Create cylinder geometries along the lines
        cylinders = []
        for i in range(len(lines)):
            start_point = np.asarray(line_set.points[lines[i][0]])
            end_point = np.asarray(line_set.points[lines[i][1]])
            direction = end_point - start_point
            cylinder = self.create_cylinder(start_point, direction)
            if cylinder is not None:
                cylinder.compute_vertex_normals()
                cylinder.paint_uniform_color([1, 0, 0])
                cylinders.append(cylinder)

        return line_set, cylinders
    
    def extract_skeletons_parallel(self, alpha=1.15, scale_factor=100):
        unique_labels = np.unique(self.labels)
        num_cores = os.cpu_count()
        input_cloud_np = np.asarray(self.input_cloud.points)

        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Using executor.submit instead of executor.map
            futures = [executor.submit(self.extract_skeleton, label, alpha, scale_factor, input_cloud_np) for label in unique_labels]
            results = [future.result() for future in futures]

        # Retrieve and process results
        results = [future.result() for future in futures]

        # Initialize list for valid skeletons
        valid_skeletons = []

        # Iterate over results and convert valid numpy arrays to Open3D point clouds
        for skeleton_np in results:
            if skeleton_np is not None and skeleton_np.size > 0:
                # Convert numpy array to Open3D point cloud
                skeleton_pcd = o3d.geometry.PointCloud()
                skeleton_pcd.points = o3d.utility.Vector3dVector(skeleton_np)
                skeleton_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red color for skeletons
                valid_skeletons.append(skeleton_pcd)

        # Visualize the skeletons with the original point cloud
        if valid_skeletons:
            o3d.visualization.draw_geometries([self.input_cloud_copy, *valid_skeletons])
            o3d.visualization.draw_geometries([self.input_cloud_copy, self.input_cloud, *valid_skeletons, ])
        else:
            print("No valid skeletons to visualize.")

        # return valid_skeletons

        
    def extract_skeleton(self, label, alpha, scale_factor, input_cloud_np):
        # Extract each cluster using the label and input_cloud_np
        cluster_mask = self.labels == label
        cluster_points = input_cloud_np[cluster_mask]
        
        # Initialize skeleton_3d as None or as an empty array
        skeleton_3d = None

        # Perform PCA and transformation on numpy array
        mean_point = np.mean(cluster_points, axis=0)
        centered_points = cluster_points - mean_point
        covariance_matrix = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        rotated_points = np.dot(centered_points, eigenvectors)

        # Project to XY plane
        xy_points = rotated_points[:, :2]

        # Generate alpha shape
        alpha_shape = alphashape.alphashape(xy_points, alpha)

        # Create a binary image for alpha shape
        min_x, min_y = np.min(xy_points, axis=0)
        max_x, max_y = np.max(xy_points, axis=0)
        image_width = int((max_x - min_x) * scale_factor)
        image_height = int((max_y - min_y) * scale_factor)
        binary_image = np.zeros((image_height, image_width), dtype=bool)

        # Rasterize the alpha shape
        if isinstance(alpha_shape, Polygon):
            self.rasterize_polygon(alpha_shape, binary_image, scale_factor, min_x, max_y)
        elif isinstance(alpha_shape, MultiPolygon):
            for poly in alpha_shape.geoms:
                self.rasterize_polygon(poly, binary_image, scale_factor, min_x, max_y)

        # Perform skeletonization
        skeleton = morphology.skeletonize(binary_image)

        # Convert skeleton to 3D coordinates
        skeleton_indices = np.transpose(np.nonzero(skeleton))
        skeleton_xy = np.column_stack((
            skeleton_indices[:, 1] / scale_factor + min_x,
            max_y - skeleton_indices[:, 0] / scale_factor
        ))

        # Only proceed if there are skeleton points to process
        if skeleton_xy.size > 0:

            # Interpolate Z-coordinates
            # Convert the 3D points to a numpy array for sklearn compatibility
            points_3d = np.asarray(rotated_points)
            # Initialize the nearest neighbors model
            nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(points_3d[:, :2])
            _ , indices = nbrs.kneighbors(skeleton_xy)
            # Interpolate Z-coordinates
            skeleton_z = np.mean(points_3d[indices, 2], axis=1)  # This takes the average Z value of the neighbors
            skeleton_3d_pca = np.hstack((skeleton_xy, skeleton_z.reshape(-1, 1)))

            # Convert from PCA back to original point cloud coordinates
            skeleton_3d = np.dot(skeleton_3d_pca, eigenvectors.T) + mean_point

        # Return the NumPy array representation of the skeleton
        return skeleton_3d


    # Function to rasterize and fill the polygon onto the binary image
    @staticmethod
    def rasterize_polygon(poly, binary_img, scale_factor, min_x, max_y):
        if poly.is_empty:
            return
        exterior = np.array(poly.exterior.coords)
        interior_paths = [np.array(interior.coords) for interior in poly.interiors]

        # Scale and convert coordinates to image indices
        exterior[:, 0] = (exterior[:, 0] - min_x) * scale_factor
        exterior[:, 1] = (max_y - exterior[:, 1]) * scale_factor
        exteriors = [exterior]
        
        for interior in interior_paths:
            interior[:, 0] = (interior[:, 0] - min_x) * scale_factor
            interior[:, 1] = (max_y - interior[:, 1]) * scale_factor
            exteriors.append(interior)
        
        # Fill the polygon
        rr, cc = draw.polygon(exterior[:, 1], exterior[:, 0], binary_img.shape)
        binary_img[rr, cc] = True

        # Unfill the holes
        for interior in interior_paths:
            rr, cc = draw.polygon(interior[:, 1], interior[:, 0], binary_img.shape)
            binary_img[rr, cc] = False



