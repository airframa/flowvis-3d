import open3d as o3d
import numpy as np

# .py file containing utility functions

def load_pcd(data_path):
        """
        Load 3D point cloud data from the specified .ply file and return it.

        Args:
            data_path (str): The path to the .ply file containing the point cloud data.

        Returns:
            o3d.geometry.PointCloud: The loaded 3D point cloud data.
        """
        try:
            # Load the 3D point cloud data from the specified .ply file.
            point_cloud = o3d.io.read_point_cloud(data_path)

            return point_cloud
        except Exception as e:
            raise ValueError(f"Error loading data from {data_path}: {str(e)}")

def load_stl(stl_path):
    """
    Load 3D model data from the specified .stl file, apply material properties, and return it as a TriangleMesh.

    Args:
        stl_path (str): The path to the .stl file containing the 3D model data.

    Returns:
        o3d.geometry.TriangleMesh: The loaded 3D model data as a TriangleMesh with surface normals.
    """
    try:
        # Load the 3D model data from the specified .stl file.
        mesh = o3d.io.read_triangle_mesh(stl_path)

        # Apply material properties to enhance visualization.
        mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Set color

        # Compute surface normals for the mesh.
        mesh.compute_vertex_normals()

        return mesh
    except Exception as e:
        raise ValueError(f"Error loading and enhancing STL data from {stl_path}: {str(e)}")

def calculate_point_cloud_metrics(pcd):
    """
    Calculate various metrics for a given 3D point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): The input 3D point cloud.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # Calculate the number of points in the point cloud
    num_points = len(pcd.points)

    # Calculate the bounding box dimensions
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    bounding_box_dimensions = np.asarray(max_bound) - np.asarray(min_bound)

    # Calculate point cloud density (points per unit volume)
    point_density = num_points / np.prod(bounding_box_dimensions)

    # Create a dictionary to store the calculated metrics
    metrics = {
        "Number of Points": num_points,
        "Bounding Box Dimensions (X, Y, Z)": bounding_box_dimensions,
        "Point Density": point_density
    }

    return metrics

def change_file_ext(path):
        """
        Change the target path from .ply to .stl.

        Args:
            target_path (str): The original target path with .ply extension.

        Returns:
            str: The modified target path with .stl extension.
        """
        if path is not None:
            return path.replace(".ply", ".stl")
        return None


