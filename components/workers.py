from PySide6.QtCore import QObject, Signal, Slot
from components.point_cloud_registration import PointCloudRegistration
from components.point_cloud_utils import load_pcd
import open3d as o3d
import copy
import os
import tarfile
import tempfile
import shutil



class LoadPointCloudWorker(QObject):
    """
    Asynchronously loads point cloud data and emits signals on completion or error.
    """
    finished = Signal(object, str)  # Emits point cloud data or an error message.

    def __init__(self, file_path=None, parent=None):
        """
        Initializes the worker with an optional file path and parent QObject.
        """
        super().__init__(parent)
        self.file_path_pcd = file_path  # Path to the point cloud file to be loaded.

    def run(self):
        """
        Loads point cloud data from file_path, emitting finished signal on success or failure.
        Designed to run in a separate thread to keep UI responsive.
        """
        try:
            pcd = load_pcd(self.file_path_pcd)  # Attempt to load the point cloud.
            self.finished.emit(pcd, "")  # Emit loaded point cloud with no error message.
        except Exception as e:
            self.finished.emit(None, str(e))  # On error, emit None and the error message.

class LoadReferenceWorker(QObject):
    """
    Loads reference geometry data asynchronously and communicates status via signals.
    """
    dataLoaded = Signal(object)  # Emitted with the loaded geometry data.
    error = Signal(str)          # Emitted with an error message.
    finished = Signal()          # Emitted when processing is complete.
    active = False               # Indicates whether the worker is actively processing.

    def __init__(self, reference_path):
        """
        Initializes the worker with a path to the reference data.
        """
        super(LoadReferenceWorker, self).__init__()
        self.reference_path = reference_path
        self.active = True  # Set to active upon initialization.

    @Slot()
    def run(self):
        """
        Loads the reference geometry from the specified path if the worker is active.
        Emits dataLoaded or error signals as appropriate.
        """
        if not self.active:
            return  # Exit if not active.
        try:
            reference_geom = o3d.io.read_triangle_mesh(self.reference_path)  # Load mesh from file.
            reference_geom.compute_vertex_normals()  # Compute normals for the mesh.
            if self.active:  # Check again before emitting signal.
                self.dataLoaded.emit(reference_geom)  # Emit the loaded geometry data.
        except Exception as e:
            if self.active:
                self.error.emit(str(e))  # Emit an error message if still active.
        finally:
            self.active = False  # Set to inactive once processing is complete.

    def deleteLater(self):
        """
        Safely deletes the worker, ensuring it ceases operations first.
        """
        if self.active:
            self.active = False  # Disable any further activity.
        super(LoadReferenceWorker, self).deleteLater()  # Proceed with QObject deletion.

class RegistrationWorker(QObject):
    """
    Manages the asynchronous registration of point clouds and communicates results via signals.
    """
    finished = Signal(object, object, str)  # Emitted with the registered point cloud, transformation matrix, and log message.
    error = Signal(str)  # Emitted with error message.
    requestStop = Signal()  # Signal to request stopping the process.
    active = False  # Indicates whether the worker is actively processing.

    def __init__(self, sourcePath, referencePath, main_app, voxel_sizes, desired_fitness_ransac, desired_fitness_icp):
        """
        Initializes the worker with paths for source and reference point clouds and registration parameters.
        """
        super(RegistrationWorker, self).__init__()
        self.sourcePath = sourcePath
        self.referencePath = referencePath
        self.main_app = main_app  # Application context or main app object.
        self.voxel_sizes = voxel_sizes  # Voxel sizes for down-sampling.
        self.desired_fitness_ransac = desired_fitness_ransac  # Fitness threshold for RANSAC.
        self.desired_fitness_icp = desired_fitness_icp  # Fitness thresholds for ICP.
        self.active = True

    def stop(self):
        """
        Safely stops the worker if it is active.
        """
        if self.active:
            self.active = False

    @Slot()
    def run(self):
        """
        Loads point clouds, performs registration, and emits the result or error.
        Only runs if the worker is active.
        """
        if not self.active:
            return
        try:
            source_pcd = load_pcd(self.sourcePath)  # Load source point cloud.
            reference_pcd = load_pcd(self.referencePath)  # Load reference point cloud.
            registration = PointCloudRegistration(source=source_pcd, target=reference_pcd)

            # Perform registration and obtain results.
            pcd_registered, transformation, log_text = registration.register(
                voxel_sizes=self.voxel_sizes,
                desired_fitness_ransac=self.desired_fitness_ransac,
                desired_fitness_icp=self.desired_fitness_icp
            )

            if self.active:
                self.finished.emit(pcd_registered, transformation, log_text)  # Emit successful registration results.
                # print("Emitting finished signal with:", pcd_registered, transformation, log_text)

        except Exception as e:
            if self.active:
                self.error.emit(str(e))  # Emit error if the process fails.
        finally:
            self.active = False  # Mark the worker as inactive.

    def deleteLater(self):
        """
        Ensures the worker stops processing before deletion.
        """
        if self.active:
            self.requestStop.emit()
        super(RegistrationWorker, self).deleteLater()

class SaveWorker(QObject):
    """
    Handles the saving of registered point cloud data and optionally mesh data to disk.
    """
    finished = Signal()  # Emitted when the saving process completes.
    error = Signal(str)  # Emitted on error during the saving process.
    log_message = Signal(str)  # Emitted to send log messages regarding the saving status.
    requestStop = Signal()  # Signal to request stopping the process.
    active = False  # Indicates whether the worker is actively processing.

    def __init__(self, main_app, file_path, registered_pcd, transformation, composed_filename, save_mesh=False):
        """
        Initializes the worker with the application context and saving parameters.
        """
        super(SaveWorker, self).__init__()
        self.main_app = main_app
        self.file_path_pcd = file_path
        self.registered_pcd = registered_pcd
        self.transformation = transformation
        self.composed_filename = composed_filename
        self.save_mesh = save_mesh
        self.scaled_mesh = None  # Add an instance variable to store the scaled mesh
        self.active = True

    def stop(self):
        """
        Safely stops the worker if it is active.
        """
        if self.active:
            self.active = False
            self.requestStop.emit()

    @Slot()
    def run(self):
        """
        Performs the saving of the point cloud and, if specified, mesh data. Emits signals on completion or error.
        Only runs if the worker is active.
        """
        if not self.active:
            return
        
        try:
            if self.registered_pcd is not None:
                pcd_copy = copy.deepcopy(self.registered_pcd)
                directory_path = os.path.dirname(self.file_path_pcd)
                output_file_raw = os.path.join(directory_path, f"{self.composed_filename}_registered.ply")
                output_file_scaled = os.path.join(directory_path, f"{self.composed_filename}_registered_paraview.ply")
                o3d.io.write_point_cloud(output_file_raw, pcd_copy)
                scale = 1 / 600
                registered_pcd_scaled = pcd_copy.scale(scale, center=(0, 0, 0))
                o3d.io.write_point_cloud(output_file_scaled, registered_pcd_scaled)
                # Store saved file for output
                saved_files = ["original registered point cloud", "CFD-scaled registered point cloud"]

                if self.save_mesh:
                    try:
                        mesh = o3d.io.read_triangle_mesh(self.file_path_pcd)
                        if len(mesh.triangles) > 0:
                            # define output file paths
                            output_mesh_file_raw = os.path.join(directory_path, f"{self.composed_filename}_registered_mesh.ply")
                            output_mesh_file = os.path.join(directory_path, f"{self.composed_filename}_registered_mesh_paraview.ply")
                            # transform mesh
                            mesh_registered = mesh.transform(self.transformation)
                            # save unscaled mesh
                            o3d.io.write_triangle_mesh(output_mesh_file_raw, mesh_registered)
                            saved_files.append("original registered mesh")
                            # save scaled mesh
                            mesh_registered.scale(scale, center=(0, 0, 0))
                            o3d.io.write_triangle_mesh(output_mesh_file, mesh_registered)
                            saved_files.append("CFD-scaled registered mesh")
                            summary_message = "Mesh data available for the current .ply file\n"
                            # store scaled mesh for sandbox upload
                            self.scaled_mesh = mesh_registered
                        else:
                            summary_message = "No mesh data available for the current .ply file\n"
                    except Exception as e:
                        self.error.emit(str(e))
                        summary_message = "Failed to read mesh data\n"

                # Compile summary of saved files and their location.
                summary_message += f"Registered data ({', '.join(saved_files)}) saved in the directory: {directory_path}"
                self.log_message.emit(summary_message)
            else:
                self.log_message.emit("No registered point cloud available. Perform registration first.")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.active = False
            self.finished.emit()

    def deleteLater(self):
        """
        Ensures the worker stops processing before deletion.
        """
        self.stop()
        super(SaveWorker, self).deleteLater()

class UploadWorker(QObject):
    """
    Handles the upload of registered data to a sandbox environment.
    """
    finished = Signal()  # Emitted when the upload process completes.
    error = Signal(str)  # Emitted on error during the upload process.
    log_message = Signal(str)  # Emitted to send log messages regarding the upload status.
    requestStop = Signal()  # Signal to request stopping the process.
    active = False  # Indicates whether the worker is actively processing.

    def __init__(self, main_app, scaled_mesh, car_part, target_directory):
        """
        Initializes the worker with the application context and upload parameters.

        Args:
            main_app: Reference to the main application instance.
            scaled_mesh: The scaled mesh to be uploaded.
            car_part: The car part name.
            target_directory: The target directory for the upload.
        """
        super(UploadWorker, self).__init__()
        self.main_app = main_app
        self.scaled_mesh = scaled_mesh
        self.car_part = car_part
        self.target_directory = target_directory
        self.active = True

    def stop(self):
        """
        Safely stops the worker if it is active.
        """
        if self.active:
            self.active = False
            self.requestStop.emit()

    @Slot()
    def run(self):
        """
        Performs the upload of the scaled mesh. Emits signals on completion or error.
        Only runs if the worker is active.
        """
        if not self.active:
            return

        try:
            # Create temporary directory for storing the files before upload
            temp_dir = tempfile.mkdtemp()
            wt_run = self.main_app.WTRunLineEdit.text().strip()
            wt_run_folder = os.path.join(temp_dir, wt_run)
            os.makedirs(wt_run_folder)

            # Create the "distiller2" folder inside the WT Run folder
            distiller2_folder = os.path.join(wt_run_folder, "distiller2")
            os.makedirs(distiller2_folder)

            # Get the car part short name from the dictionary
            car_part_short_name = self.main_app.car_part_correspondences[self.car_part]

            # Save the registered mesh
            mesh_file_path = os.path.join(distiller2_folder, f"{car_part_short_name}.ply")
            o3d.io.write_triangle_mesh(mesh_file_path, self.scaled_mesh)

            # Compress the folder
            tar_file_path = self.compressFolder(wt_run_folder)

            # Upload the .tar file to the target directory
            self.uploadFile(tar_file_path, self.target_directory)

            # Emit the finished signal to indicate the process is complete
            self.finished.emit()

        except Exception as e:
            # Emit the error signal with the exception message if an error occurs
            self.error.emit(str(e))
        finally:
            # Set the worker's active state to False, indicating the process is no longer running
            self.active = False

    def compressFolder(self, folder_path):
        """
        Compresses the specified folder into a .tar file.

        Args:
            folder_path: The path of the folder to compress.

        Returns:
            The path of the created .tar file.
        """
        tar_file_path = folder_path + ".tar"
        with tarfile.open(tar_file_path, "w") as tar:
            tar.add(folder_path, arcname=os.path.basename(folder_path))
        return tar_file_path

    def uploadFile(self, file_path, target_directory):
        """
        Uploads the specified file to the target directory.

        Args:
            file_path: The path of the file to upload.
            target_directory: The directory where the file should be uploaded.
        """
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        
        target_path = os.path.join(target_directory, os.path.basename(file_path))
        shutil.copy(file_path, target_path)
        self.log_message.emit(f"Uploaded .tar file to: {target_path}")

    def deleteLater(self):
        """
        Ensures the worker stops processing before deletion.
        """
        self.stop()
        super(UploadWorker, self).deleteLater()
















