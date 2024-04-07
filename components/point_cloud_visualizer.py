import open3d as o3d
import ipywidgets as widgets
from IPython.display import clear_output, display, HTML
from components.point_cloud_utils import load_pcd, load_stl

class PointCloudVisualizer:
    def __init__(self, input_clouds=None, input_path=None, target_mesh=None, target_path=None):
        """
        Initialize the PointCloudVisualizer.

        Args:
            input_cloud (PointCloud, optional): An instance of the custom PointCloud class representing loaded 3D point cloud data. Lists of point clouds are also accepeted.
            input_path (str, optional): The path to the .ply file containing the input point cloud data.
            target_mesh (o3d.geometry.TriangleMesh, optional): The loaded 3D mesh data.
            target_path (str, optional): The path to the .stl file containing the target 3D model data.
        """
        # Store the provided input and target paths.
        self.input_path = input_path
        self.target_path = target_path  

        # Load the input point cloud if provided, otherwise load from the input path.
        # Store the provided input clouds.
        if isinstance(input_clouds, o3d.geometry.PointCloud):
            self.input_clouds = [input_clouds]  # Make it a list if it's a single point cloud
        elif isinstance(input_clouds, list):
            self.input_clouds = input_clouds
        elif input_path is not None:
            self.input_cloud = load_pcd(input_path)
        else:
            self.input_clouds = []

        # Load the target 3D model if provided, otherwise load from the target path.
        if target_mesh is not None:
            self.target_mesh = target_mesh
        elif target_path is not None:
            self.target_mesh = load_stl(target_path)
        else:
            self.target_mesh = None

        # compute normals for visualization purposes
        # self.input_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
            

        # Create a button for visualization with custom style.
        self.show_button = widgets.Button(description="Show 3D data")
        self.show_button.layout.button_color = 'black'
        self.show_button.style.button_color = 'red'
        self.show_button.style.font_weight = 'bold'
        
        # Output area for displaying the visualization.
        self.output = widgets.Output()

        # Attach button click handler.
        self.show_button.on_click(self.show_data)
        
        # Apply custom styles for output background.
        custom_style = """
        <style>
        .cell-output-ipywidget-background {
           background-color: transparent !important;
        }
        .jp-OutputArea-output {
           background-color: transparent;
        }
        </style>
        """
        display(HTML(custom_style))
        
        # Display the button and the output area.
        display(self.show_button, self.output)

    def show_data(self, b):
        """
        Callback function for the "Show" button click.

        Args:
            b (ipywidgets.Button): The button widget.
        """
        with self.output:
            clear_output(wait=True)
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            if self.target_mesh is not None:
                # Remove the target mesh
                vis.remove_geometry(self.target_mesh)

            if self.input_clouds is not None:
                # Add the input point cloud
                for input_cloud in self.input_clouds:
                    vis.add_geometry(input_cloud)

            if self.target_mesh is not None:
                # Re-add the target mesh
                vis.add_geometry(self.target_mesh)
                
            # Customize the point size
            render_option = vis.get_render_option()
            render_option.point_size = 3.0  # Adjust the point size as needed
            render_option.point_show_normal = False

            vis.run()
            vis.destroy_window()


# Example usage:
# if __name__ == "__main__":
#     Use the class method to create an instance with input and target paths
#     visualizer = PointCloudVisualizer(input_path="path_to_input_cloud.ply", target_path="path_to_target_model.stl")
