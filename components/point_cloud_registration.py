import open3d as o3d
import numpy as np
import copy
import tempfile
import os
from components.point_cloud_visualizer import PointCloudVisualizer
from components.point_cloud_preprocessor import PointCloudPreprocessor
from components.point_cloud_utils import change_file_ext


class PointCloudRegistration(PointCloudPreprocessor):
    def __init__(self, source, target):
        """
        Initialize the PointCloudRegistration class.

        Args:
            source (str or o3d.geometry.PointCloud): Either the path to the source .ply file containing the point cloud data
                or the loaded source point cloud itself.
            target (str or o3d.geometry.PointCloud): Either the path to the target .ply/.stl file containing the point cloud data
                or the loaded target point cloud itself.
        """

        self.registered_pcd = None  # Initialize the registered_pcd attribute
        
        # Inheritance
        super().__init__(source)  # Inherit from the preprocessing class, no point cloud normals calculation is required in this step

        # Initialize the class with either point cloud paths or point clouds
        if isinstance(source, str):
            self.source_pcd = o3d.io.read_point_cloud(source)
        else:
            self.source_pcd = source

        if isinstance(target, str):
            # obtain point cloud either from .ply or .stl file
            if target.endswith(".ply"):
                self.target_pcd = o3d.io.read_point_cloud(target)
            elif target.endswith(".stl"):
                self.target_pcd = self.stl_to_point_cloud(target)
            else:
                log_text = "Please select a .ply or .stl file"
                return log_text
            self.target_path = target
        else:
            self.target_pcd = target
            self.target_path = None

        # Create a log file path including the file name and extension
        self.log_file_path = "//spe-ch-md9/data/Departments/Aerodynamics/Development/FlowViz/FV_CFD_REF/registration_log.txt"
        # Then, open the log file for writing as before
        self.log_file = open(self.log_file_path, "w")


        # Initialize global registration transfromation variable
        self.transformation = None

    def stl_to_point_cloud(self, stl_file_path):
        """
        Converts an STL file to a Point Cloud and visualizes it.

        Args:
            stl_file_path (str): Path to the STL file.

        Returns:
            o3d.geometry.PointCloud: The converted point cloud.
        """
        # Read the mesh from an STL file
        mesh = o3d.io.read_triangle_mesh(stl_file_path)

        # Create a temporary directory to save the intermediate PLY file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_ply_path = os.path.join(temp_dir, 'temp_mesh.ply')
            
            # Save the mesh to a PLY file
            o3d.io.write_triangle_mesh(temp_ply_path, mesh)
            
            # Read the PLY file back in as a point cloud
            pcd = o3d.io.read_point_cloud(temp_ply_path)
            
            # No need to manually delete the temp file; it will be removed with the temporary directory
            return pcd

    def preprocess_point_cloud(self, pcd, voxel_size):
        """
        Preprocess a point cloud by downsampling and computing features.

        Args:
            pcd (o3d.geometry.PointCloud): The input point cloud.
            voxel_size (float): The voxel size for downsampling.

        Returns:
            o3d.geometry.PointCloud: The downsampled point cloud.
            o3d.pipelines.registration.Feature: The computed FPFH feature.
        """
        # Downsample the point cloud using voxel grid downsampling
        pcd_down = pcd.voxel_down_sample(voxel_size)

        # Estimate normals for the downsampled point cloud
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # Compute Fast Point Feature Histograms (FPFH) for feature-based registration
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        return pcd_down, pcd_fpfh

    def prepare_dataset(self, source, target, voxel_size):
        """
        Prepare the source and target point clouds for registration.

        Args:
            source (o3d.geometry.PointCloud): The source point cloud.
            target (o3d.geometry.PointCloud): The target point cloud.
            voxel_size (float): The voxel size for downsampling.

        Returns:
            o3d.geometry.PointCloud: The source point cloud.
            o3d.geometry.PointCloud: The target point cloud.
            o3d.geometry.PointCloud: The downsampled source point cloud.
            o3d.geometry.PointCloud: The downsampled target point cloud.
            o3d.pipelines.registration.Feature: The computed FPFH feature for the source.
            o3d.pipelines.registration.Feature: The computed FPFH feature for the target.
        """
        # Initialize the transformation matrix for source point cloud
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        source.transform(trans_init)

        # Preprocess the source and target point clouds
        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)

        return source, target, source_down, target_down, source_fpfh, target_fpfh

    def execute_global_registration(self, source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        """
        Execute global registration using RANSAC-based feature matching.

        Args:
            source_down (o3d.geometry.PointCloud): The downsampled source point cloud.
            target_down (o3d.geometry.PointCloud): The downsampled target point cloud.
            source_fpfh (o3d.pipelines.registration.Feature): The FPFH feature for the source.
            target_fpfh (o3d.pipelines.registration.Feature): The FPFH feature for the target.
            voxel_size (float): The voxel size for downsampling.

        Returns:
            o3d.pipelines.registration.RegistrationResult: The registration result.
        """
        # Set the distance threshold for RANSAC-based registration
        distance_threshold = voxel_size * 1.5

        # Execute RANSAC-based registration
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

        return result

    def refine_registration(self, source, target, voxel_size, transformation):
        """
        Refine registration using point-to-plane ICP.

        Args:
            source (o3d.geometry.PointCloud): The source point cloud.
            target (o3d.geometry.PointCloud): The target point cloud.
            voxel_size (float): The voxel size for downsampling.
            transformation (numpy.ndarray): The initial transformation matrix.

        Returns:
            o3d.pipelines.registration.RegistrationResult: The refined registration result.
        """
        # Set the distance threshold for ICP registration
        distance_threshold = voxel_size * 0.4

        # Estimate normals for source and target point clouds
        radius_normal = 1.0 * 2
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # Refine the registration using point-to-plane ICP
        result = o3d.pipelines.registration.registration_icp(
           source, target, distance_threshold, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        return result

    def register(self, voxel_sizes=[20.0, 25.0, 30.0], desired_fitness_ransac=0.90, desired_fitness_icp=[0.65, 0.70, 0.75]):
        """
        Register the source point cloud to the target point cloud with multiple voxel sizes and dynamic ICP fitness thresholds.

        Args:
            voxel_sizes (list of float, optional): A list of voxel sizes to be used for downsampling the point cloud and detect its features during registration. The default is [20.0, 25.0, 30.0], chosen based on the scale of the problem at hand.
            desired_fitness_ransac (float, optional): The desired RANSAC fitness threshold used for global registration. The default is 0.90.
            desired_fitness_icp (list of float, optional): A list of desired ICP fitness thresholds for each registration iteration. The default is [0.60, 0.70, 0.80], specifying different fitness levels for the ICP refinement stage.

        Returns:
            o3d.geometry.PointCloud: The final registered point cloud obtained after the registration process.
        """
        # Check if both source and target point clouds are provided
        if not hasattr(self, 'source_pcd') or not hasattr(self, 'target_pcd'):
            raise ValueError("Both source and target point clouds must be provided.")
        
        # Pre-compute the preprocessed point cloud
        pcd_prepr = self.preprocess_data(point_density_threshold=2.0, normals=False, normals_check=False, save=False, output=False)

        # Initialize variables to store the best registration result and provide log to the user
        icp = 0
        best_registration = None
        best_icp_fitness = 0.0
        best_ransac_fitness = 0.0  # Initialize to zero
        best_voxel_size = None
        icp_fitness_values = []

        # Iterate over different voxel sizes
        for voxel_size, desired_fitness_icp_value in zip(voxel_sizes, desired_fitness_icp):
            self.log_file.write(f"Trying voxel size: {voxel_size}\n")
            # Initialize variables to keep track of the registration process
            iterations_ransac = 0
            max_iterations_ransac = 75  # Set a maximum number of iterations to avoid infinite loops

            # RANSAC Global Registration
            self.log_file.write(":: RANSAC registration on heavily downsampled point clouds: iteration until fitness >= %.2f is achieved\n" % desired_fitness_ransac)
            while iterations_ransac < max_iterations_ransac:
                self.log_file.write(f"Iteration {iterations_ransac + 1}/{max_iterations_ransac}\n")

                # Prepare the source and target point clouds for registration
                source_down, source_fpfh = self.preprocess_point_cloud(self.source_pcd, voxel_size)
                target_down, target_fpfh = self.preprocess_point_cloud(self.target_pcd, voxel_size)

                # Execute RANSAC-based registration
                result_ransac = self.execute_global_registration(source_down, target_down,
                                                                source_fpfh, target_fpfh,
                                                                voxel_size)
                self.log_file.write("RANSAC Global Registration fitness: %.3f.\n" % result_ransac.fitness)

                # Store the best RANSAC result (if it's better)
                if result_ransac.fitness > best_ransac_fitness:
                    best_ransac_fitness = result_ransac.fitness
                    best_registration = result_ransac
                    best_voxel_size = voxel_size

                # If the desired RANSAC quality is met, break the loop
                if result_ransac.fitness >= desired_fitness_ransac:
                    self.log_file.write("RANSAC Registration quality meets the desired threshold. Proceed with ICP for fine alignment\n")
                    break

                # If the desired quality is not met, continue the loop
                iterations_ransac += 1

            # Check if the desired RANSAC fitness is achieved before proceeding with ICP
            if result_ransac.fitness >= desired_fitness_ransac:
                # Visualize the final registration result after RANSAC
                # self.draw_registration_result(self.source_pcd, self.target_pcd, result_ransac.transformation) -> display remove from released class since it requires user interaction to manually close the display window
                
                # counter to check if ICP fine registration is executed (log purposes for the end user)
                icp =+ 1

                # Refine the registration using ICP
                self.log_file.write(":: Point-to-plane ICP registration is applied on downsampled point cloud to refine the alignment\n")
                result_icp = self.refine_registration(pcd_prepr, self.target_pcd, voxel_size, result_ransac.transformation)


                # Visualize the final registration result after ICP
                # self.draw_registration_result(self.source_pcd, self.target_pcd, result_icp.transformation) -> display remove from released class since it requires user interaction to manually close the display window

                # Log result_icp into the log file
                self.log_file.write("ICP Fine Registration fitness: %.3f.\n" % result_icp.fitness)
                self.log_file.write("ICP Transformation Matrix:\n")
                self.log_file.write(str(result_icp.transformation))
                self.log_file.write("\n")

                # Store the RANSAC and corresponding ICP fitness value for this iteration
                icp_fitness_values.append(result_icp.fitness)

                # Update the best registration if the ICP fitness is better, and store also the corresponding RANSAC fitness and its voxel size for log purposes
                if result_icp.fitness > best_icp_fitness:
                    best_icp_fitness = result_icp.fitness
                    best_registration = result_icp
                    best_ransac_fitness = result_ransac.fitness
                    best_voxel_size = voxel_size

                # If the desired ICP fitness is met, stop iterating
                if best_icp_fitness >= desired_fitness_icp_value:
                    self.log_file.write("Desired ICP fitness threshold met. Stopping further iterations.\n")
                    break

            else:
                # keep track of missed icp iteration for log and user recommendation purposes
                icp_fitness_values.append(0.0)
            

        # Close the log file
        self.log_file.close()

        # Print the final transformation data and metrics to the console
        # log_text = "\n"
        path = "//srvnetapp00/Technical/Aerodynamics/Development/FlowViz/FV_CFD_REF"
        log_text = "Registration procedure completed. Check the procedure log in the file 'registration_log.txt' in the folder at '{}'.".format(path)
        log_text += f"\nRANSAC Global Registration fitness: {best_ransac_fitness}\n"
        # log_text += f"Voxel size for feature detection: {best_voxel_size}\n"
        # log_text += f"Fitness: {best_ransac_fitness}\n"
        if icp > 0:
            log_text += f"Final ICP Fine Registration fitness: {best_registration.fitness}\n"
            # log_text += f"Fitness: {best_registration.fitness}\n"
            log_text += f"Transformation matrix:\n{best_registration.transformation}"
        else:
             log_text += f"Transformation matrix:\n{best_registration.transformation}\n"
             log_text += "Final ICP Fine Registration not executed due to poor RANSAC fitness results\n"

        # Check if all ICP fitness values are below their respective desired thresholds
        if all(icp_fitness < desired_fitness for icp_fitness, desired_fitness in zip(icp_fitness_values, desired_fitness_icp)):
            pass
            # print("Registration did not meet the desired accuracy standards. We recommend repeating the registration.")

        # return registered point cloud
        self.registered_pcd = self.source_pcd.transform(best_registration.transformation)
        # return transformation
        self.transformation = best_registration.transformation

        # Enable registration visualization
        # # log_text += "Registered point cloud:"
        # PointCloudVisualizer(input_clouds=self.registered_pcd, target_path=change_file_ext(self.target_path))

        return self.registered_pcd, best_registration.transformation, log_text # Return the registered point cloud and the transfromation matrix
    
    def save_registered_ply(self, file_path, save_mesh = False):
        """
        Save the registered point cloud to a .ply file with the new name. Optionally, the user can also store the registered point cloud in mesh format (using again .ply format).

        Args:
            file_path (str): The path of the original point cloud file.
            
        Returns:
            None
        """
        # Check if there is a registered point cloud available for saving
        if self.registered_pcd is not None:
            # Create a new file path with "_registered.ply" appended to the original file name
            output_file_raw = file_path.replace('.ply', '_registered.ply')
            output_file_scaled = file_path.replace('.ply', '_registered_paraview.ply')
            # Use Open3D's IO module to write the point cloud to the new .ply file
            o3d.io.write_point_cloud(output_file_raw, self.registered_pcd)
            # Scale point cloud for paraview setting
            scale = 1 / 600  # scale factor from mm to m (1/1000) +scale factor MS to FS (1000/600) -> match sandbox units + scale
            # Apply scaling to the mesh
            self.registered_pcd.scale(scale, center=(0, 0, 0))
            # Use Open3D's IO module to write the point cloud to the new .ply file
            o3d.io.write_point_cloud(output_file_scaled, self.registered_pcd)
            # Print a success message with the file path where the point cloud is saved
            print(f"Registered point cloud saved to {output_file_raw}")
            print(f"Registered scaled point cloud saved to {output_file_scaled}")
        else:
            # If there's no registered point cloud, print a message indicating that registration must be performed first
            print("No registered point cloud available. Perform registration first.")

        # save registered mesh if desired
            # this step is intedended to export the registered mesh .ply file in Sandbox
        if save_mesh:
            try:
                # read mesh file (generated by Einstar scanner)
                mesh = o3d.io.read_triangle_mesh(file_path)
                # apply transformation
                mesh_registered = mesh.transform(self.transformation)
                # Scale mesh from mm to m to fit Sandbox data units
                # Create a scaling matrix
                scale = 1 / 600  # scale factor from mm to m (1/1000) +scale factor MS to FS (1000/600) -> match sandbox units + scale
                # Apply scaling to the mesh
                mesh_registered.scale(scale, center=(0, 0, 0))
                # Save the scaled and registered mesh
                output_file = file_path.replace('.ply', '_registered_mesh_paraview.ply')
                # use Open3D's IO module to write the point cloud to the new .ply file
                o3d.io.write_triangle_mesh(output_file, mesh_registered)
                # Print a success message with the file path where the point cloud is saved
                print(f"Registered mesh saved to {output_file}")
            except Exception as e:
                # If there's no mesh information, print a message indicating that
                print(f"No mesh information in the {file_path} file is available.")



    def draw_registration_result(self, source, target, transformation):
        """
        Visualize the registration result by rendering the source and target point clouds after transformation.

        Args:
            source (o3d.geometry.PointCloud): The source point cloud.
            target (o3d.geometry.PointCloud): The target point cloud.
            transformation (np.ndarray): The transformation matrix aligning the source to the target.

        Returns:
            None
        """
        # Create deep copies of the source and target point clouds to avoid modifying the original data
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        # Set colors for visualization
        source_temp.paint_uniform_color([1, 0.706, 0])  # Source point cloud color (orange)
        target_temp.paint_uniform_color([0, 0.651, 0.929])  # Target point cloud color (blue)

        # Apply the transformation to the source point cloud
        source_temp.transform(transformation)

        # Define visualization parameters (zoom, view direction, and up direction)
        zoom = 0.4559
        front = [0.6452, -0.3036, -0.7011]
        lookat = [1.9892, 2.0208, 1.8945]
        up = [-0.2779, -0.9482, 0.1556]

        # Render the source and target point clouds together for visualization
        o3d.visualization.draw_geometries([source_temp, target_temp], zoom=zoom, front=front, lookat=lookat, up=up)