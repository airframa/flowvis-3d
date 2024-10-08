from PySide6.QtCore import QObject, Signal, Slot, QThread
from components.point_cloud_registration import PointCloudRegistration
from components.point_cloud_utils import load_pcd
import open3d as o3d
import copy
import os
import tarfile
import tempfile
import shutil
import xml.etree.ElementTree as ET
import hashlib

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

class LoadMeshWorker(QObject):
    """
    Asynchronously loads mesh data and emits signals on completion or error.
    """
    finished = Signal(object, str)  # Emits mesh data or an error message.

    def __init__(self, file_path=None, scale=None, parent=None):
        """
        Initializes the worker with an optional file path, scale, and parent QObject.
        """
        super().__init__(parent)
        self.file_path = file_path  # Path to the initial point cloud file.
        self.scale = scale  # Scale selection (WT or CFD).

    def run(self):
        """
        Loads mesh data based on the initial file path and scale, emitting finished signal on success or failure.
        Designed to run in a separate thread to keep UI responsive.
        """
        try:
            # Attempt to load the mesh
            mesh = o3d.io.read_triangle_mesh(self.file_path)
            # Check if the mesh contains any triangles
            if len(mesh.triangles) == 0:
                raise ValueError("The loaded file does not contain mesh data, only point cloud vertices.")

            # Scale the mesh if the scale selection is CFD (100%)
            if self.scale == "CFD":
                mesh_scaled = copy.deepcopy(mesh)
                mesh_scaled.scale(600, center=(0, 0, 0))
                mesh = mesh_scaled

            # Emit the loaded (and possibly scaled) mesh with no error message
            self.finished.emit(mesh, "")
        except Exception as e:
            # On error, emit None and the error message
            self.finished.emit(None, str(e))

class RegistrationWorker(QObject):
    """
    Manages the asynchronous registration of point clouds and communicates results via signals.
    """
    finished = Signal(object, object, str, object, object, object)  # Updated signal to include additional outputs
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

            # Scale registered pcd to CFD coordinates
            pcd_registered_scaled = copy.deepcopy(pcd_registered)
            pcd_registered_scaled.scale(1 / 600, center=(0, 0, 0))  # Scale the registered point cloud

            # Initialize variables for mesh data
            registered_mesh = None
            registered_mesh_scaled = None

            # Try to read and process the mesh
            try:
                mesh = o3d.io.read_triangle_mesh(self.sourcePath)
                if len(mesh.triangles) > 0:
                    registered_mesh = mesh.transform(transformation)
                    # Scale registered mesh
                    registered_mesh_scaled = copy.deepcopy(registered_mesh)
                    registered_mesh_scaled.scale(1 / 600, center=(0, 0, 0))
                    log_text += "\nMesh data available for the current .ply file."
                else:
                    log_text += "\nNo mesh data available for the current .ply file."
            except Exception as e:
                log_text += f"\nFailed to read mesh data: {str(e)}"
                log_text += "\nUpload to Sandbox is not possible without mesh data."

            if self.active:
                self.finished.emit(pcd_registered, transformation, log_text, pcd_registered_scaled, registered_mesh, registered_mesh_scaled)  # Emit successful registration results with additional outputs

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

    def __init__(self, main_app, file_path, registered_pcd, registered_pcd_scaled, registered_mesh, registered_mesh_scaled, composed_filename):
        """
        Initializes the worker with the application context and saving parameters.
        """
        super(SaveWorker, self).__init__()
        self.main_app = main_app
        self.file_path_pcd = file_path
        self.registered_pcd = registered_pcd
        self.registered_pcd_scaled = registered_pcd_scaled
        self.registered_mesh = registered_mesh
        self.registered_mesh_scaled = registered_mesh_scaled
        self.composed_filename = composed_filename
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
            directory_path = os.path.dirname(self.file_path_pcd)
            saved_files = []

            if self.registered_pcd is not None:
                output_file_raw = os.path.join(directory_path, f"{self.composed_filename}_registered.ply")
                output_file_scaled = os.path.join(directory_path, f"{self.composed_filename}_registered_paraview.ply")
                o3d.io.write_point_cloud(output_file_raw, self.registered_pcd)
                o3d.io.write_point_cloud(output_file_scaled, self.registered_pcd_scaled)
                saved_files.extend(["original registered point cloud", "CFD-scaled registered point cloud"])

            if self.registered_mesh is not None:
                output_mesh_file_raw = os.path.join(directory_path, f"{self.composed_filename}_registered_mesh.ply")
                output_mesh_file = os.path.join(directory_path, f"{self.composed_filename}_registered_mesh_paraview.ply")
                o3d.io.write_triangle_mesh(output_mesh_file_raw, self.registered_mesh)
                o3d.io.write_triangle_mesh(output_mesh_file, self.registered_mesh_scaled)
                saved_files.extend(["original registered mesh", "CFD-scaled registered mesh"])

            summary_message = f"Registered data ({', '.join(saved_files)}) saved in the directory: {directory_path}"
            self.log_message.emit(summary_message)

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

    def __init__(self, main_app, mesh, car_part, target_directory, wt_run, case_description, map_conversion_name, case_number):
        """
        Initializes the worker with the application context and upload parameters.

        Args:
            main_app: Reference to the main application instance.
            mesh: The mesh to be uploaded (WT coordinates). The scaling to CFD coordinates is performed in SandBox.
            car_part: The car part name.
            target_directory: The target directory for the upload.
            wt_run: The WT run number.
            case_description: The case description.
            map_conversion_name: The map conversion name.
            case_number: The case number.
        """
        super(UploadWorker, self).__init__()
        self.main_app = main_app
        self.mesh = mesh
        self.car_part = car_part
        self.target_directory = target_directory
        self.wt_run = wt_run
        self.case_description = case_description
        self.map_conversion_name = map_conversion_name
        self.case_number = case_number
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
            wt_run_folder = os.path.join(temp_dir, self.wt_run)
            os.makedirs(wt_run_folder)

            # Create the "distiller2" folder inside the WT Run folder
            distiller2_folder = os.path.join(wt_run_folder, "distiller2")
            os.makedirs(distiller2_folder)

            # Get the car part short name from the dictionary
            car_part_short_name = self.main_app.car_part_correspondences[self.car_part]

            # Save the registered mesh inside the "flowviz" folder in "distiller2"
            flowviz_folder = os.path.join(distiller2_folder, "flowviz")
            os.makedirs(flowviz_folder)
            mesh_file_path = os.path.join(flowviz_folder, f"{car_part_short_name}.ply")
            o3d.io.write_triangle_mesh(mesh_file_path, self.mesh)

            # Load and modify the data.xml template
            template_path = os.path.join(os.path.dirname(__file__), '../templates/data.xml')
            tree = ET.parse(template_path)
            root = tree.getroot()
            thread = root.find(".//vtu-threads/thread")
            thread.set("name", car_part_short_name)
            thread.set("regExp", f"{car_part_short_name}.vtu")

            # Save the modified data.xml file in "distiller2"
            xml_file_path = os.path.join(distiller2_folder, "data.xml")
            tree.write(xml_file_path)

            # Modify and save the iLaunchData.xml file
            self.modify_and_save_ilauchdata(wt_run_folder)

            # Copy the additional XML files to the WT Run folder
            self.copyTemplateFiles(wt_run_folder)

            # Modify and save the post-c44-v4.2.xml file
            self.modify_and_save_post_c44(wt_run_folder)

            # Compress the folder
            tar_file_path = self.compressFolder(wt_run_folder)

            # Calculate the checksum and create the checksum XML file
            self.createChecksumXML(tar_file_path, wt_run_folder)

            # Upload the .tar file to the target directory
            self.uploadFile(tar_file_path, self.target_directory)

            # Apply a 3 s delay to make sure that the .tar file is uploaded in advance with respect to the corresponding .xml file
            QThread.sleep(3)

            # Upload the checksum XML file to the target directory
            checksum_xml_path = os.path.join(wt_run_folder, f"{self.wt_run}.xml")
            self.uploadFile(checksum_xml_path, self.target_directory)

            # Declare successful upload operation
            self.log_message.emit(f"Successful Sandbox Upload")

            # Emit the finished signal to indicate the process is complete
            self.finished.emit()

        except Exception as e:
            # Emit the error signal with the exception message if an error occurs
            self.error.emit(str(e))
        finally:
            # Set the worker's active state to False, indicating the process is no longer running
            self.active = False

    def copyTemplateFiles(self, wt_run_folder):
        """
        Copies the iLaunchData.xml and post-c44-v4.2.xml files from the templates folder to the WT Run folder.

        Args:
            wt_run_folder (str): The path of the WT Run folder where the files should be copied.
        """
        template_files = ["post-c44-v4.2.xml"]
        template_folder = os.path.join(os.path.dirname(__file__), '../templates')

        for file_name in template_files:
            src_file = os.path.join(template_folder, file_name)
            dest_file = os.path.join(wt_run_folder, file_name)
            shutil.copy(src_file, dest_file)

    def modify_and_save_ilauchdata(self, wt_run_folder):
        """
        Reads the iLaunchData.xml template, modifies the car model, name, run-description, project-label, project, run-label, and run,
        and saves it to the WT run folder.

        Args:
            wt_run_folder (str): The path of the WT run folder where the modified XML file should be saved.
        """
        template_path = os.path.join(os.path.dirname(__file__), '../templates/iLaunchData.xml')
        tree = ET.parse(template_path)
        root = tree.getroot()

        wt_model = self.main_app.modelLineEdit.text().strip()
        wt_model_number = wt_model[1:]  # Extract the number part from the WT model string

        # Modify the car element
        car_element = root.find(".//car")
        if car_element is not None:
            car_element.set("model", wt_model)
            car_element.set("name", f"C{wt_model_number}")

        # Modify the run-description element
        car_part = self.main_app.carPartComboBox.currentText().title()  # Capitalize the first letter of each word
        run_description_element = root.find(".//run-description")
        if run_description_element is not None:
            run_description_element.set("value", f"{car_part} flowviz test")

        # Modify the project-label element in parent-run
        project_label_element = root.find(".//parent-run/project-label")
        if project_label_element is not None:
            project_label_element.set("value", f"C{wt_model_number}-WT00")

        # Modify the run-label element in parent-run
        run_label_element = root.find(".//parent-run/run-label")
        if run_label_element is not None:
            run_label_element.set("value", self.wt_run)

        # Modify the project element in case-info
        case_project_element = root.find(".//case-info/labels/project")
        if case_project_element is not None:
            case_project_element.set("value", f"C{wt_model_number}-WT00")

        # Modify the run element in case-info
        case_run_element = root.find(".//case-info/labels/run")
        if case_run_element is not None:
            case_run_element.set("value", f"{self.wt_run}-FLOWVIZ")

        # Modify the case and case-description elements in case-info
        case_element = root.find(".//case-info/labels/case")
        if case_element is not None:
            case_element.set("value", self.case_number)

        case_description_element = root.find(".//case-info/case-description")
        if case_description_element is not None:
            case_description_element.set("value", self.case_description)

        xml_file_path = os.path.join(wt_run_folder, "iLaunchData.xml")
        tree.write(xml_file_path)

    def modify_and_save_post_c44(self, wt_run_folder):
        """
        Reads the post-c44-v4.2.xml template, modifies the wind-tunnel-map-conversion name,
        and saves it to the WT run folder.

        Args:
            wt_run_folder (str): The path of the WT run folder where the modified XML file should be saved.
        """
        post_c44_path = os.path.join(wt_run_folder, "post-c44-v4.2.xml")
        tree = ET.parse(post_c44_path)
        root = tree.getroot()

        # Modify the wind-tunnel-map-conversion element
        map_conversion_element = root.find(".//wind-tunnel-map-conversion")
        if map_conversion_element is not None:
            map_conversion_element.set("name", self.map_conversion_name)

        tree.write(post_c44_path)

    def compressFolder(self, folder_path):
        """
        Compresses the specified folder into a .tar file.

        Args:
            folder_path (str): The path of the folder to compress.

        Returns:
            str: The path of the created .tar file.
        """
        tar_file_path = folder_path + ".tar"
        with tarfile.open(tar_file_path, "w") as tar:
            tar.add(folder_path, arcname=os.path.basename(folder_path))
        return tar_file_path

    def createChecksumXML(self, tar_file_path, wt_run_folder):
        """
        Creates a checksum XML file for the specified tar file and saves it in the WT run folder.

        Args:
            tar_file_path (str): The path of the tar file to create the checksum for.
            wt_run_folder (str): The path of the WT run folder where the checksum XML file should be saved.
        """
        checksum = self.calculateChecksum(tar_file_path)
        checksum_xml_path = os.path.join(wt_run_folder, f"{self.wt_run}.xml")
        self.writeChecksumXML(checksum_xml_path, checksum, tar_file_path)

    def calculateChecksum(self, file_path):
        """
        Calculates the SHA1 checksum for the specified file.

        Args:
            file_path (str): The path of the file to calculate the checksum for.

        Returns:
            str: The calculated checksum.
        """
        d = hashlib.sha1()
        with open(file_path, "rb") as f:
            d.update(f.read())
        return d.hexdigest()

    def writeChecksumXML(self, xml_path, checksum, tar_file_path):
        """
        Writes the checksum and tar file information to an XML file.

        Args:
            xml_path (str): The path of the XML file to write to.
            checksum (str): The calculated checksum.
            tar_file_path (str): The path of the tar file.
        """
        # Construct XML string with proper indentation and formatting
        xml_content = f"""<sandbox>
 <tarfile value="{os.path.basename(tar_file_path)}" />
 <checksums sha1sum="{checksum}" />
 <half-car-dataset value="false" />
 <data-type value="flowviz" />
</sandbox>
"""
        # Write the XML content to the file
        with open(xml_path, "w", encoding="UTF-8") as xml_file:
            xml_file.write(xml_content)

    def uploadFile(self, file_path, target_directory):
        """
        Uploads the specified file to the target directory.

        Args:
            file_path (str): The path of the file to upload.
            target_directory (str): The directory where the file should be uploaded.
        """
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        
        target_path = os.path.join(target_directory, os.path.basename(file_path))
        shutil.copy(file_path, target_path)
        self.log_message.emit(f"Uploaded {os.path.basename(file_path)} to: {target_path}")

    def deleteLater(self):
        """
        Ensures the worker stops processing before deletion.
        """
        self.stop()
        super(UploadWorker, self).deleteLater()











