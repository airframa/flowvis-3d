from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QPushButton, QLineEdit, QFileDialog, QLabel, QTextEdit,  QMessageBox
from PySide6.QtGui import QFont, QPixmap, QDragEnterEvent, QDropEvent, QTextOption, QGuiApplication
from PySide6.QtCore import Qt, QThread
from components.utils import CollapsibleSection, SettingsSection, applyButtonStyle, applyLineEditStyle, applyTextAndScrollBarStyle, createLoadingWheel, startLoadingAnimation, stopLoadingAnimation
from components.workers import LoadPointCloudWorker, LoadReferenceWorker, RegistrationWorker, SaveWorker
import open3d as o3d
import os

class MainApp(QMainWindow):
    """
    MainApp class that extends QMainWindow, providing the main window for the application.

    This class handles the initialization and layout of the application's user interface,
    including setting up the window title, enabling drag-and-drop functionality, and configuring initial state variables and UI components.
    """

    def __init__(self):
        """
        Constructor for the MainApp class.
        """
        super(MainApp, self).__init__()  # Call the constructor of the parent class QMainWindow.
        self.setWindowTitle("FlowVis3D")  # Set the title of the main window.
        self.setAcceptDrops(True)  # Enable the main window to accept drag-and-drop events.
        self.initVariables()  # Initialize application-specific variables and state.
        self.setupUI()  # Set up the graphical user interface elements of the application.


    def initVariables(self):
        """
        Initialize variables related to application state and workers.
        """
        self.registrationThread = None
        self.registrationWorker = None
        self.visualizationThread = None
        self.LoadReferenceWorker = None
        self.saveThread = None
        self.saveWorker = None
        self.file_path_pcd = None  
        self.reference_geometry = None
        self.cached_reference_path = None
        self.cached_pcd = None
        self.cached_pcd_path = None
        self.registered_pcd = None
        self.transformation = None
        self.workerIsActive = False

    def setupUI(self):
        """
        Set up the user interface of the main application window.
        """
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)
        self.setupLogo(layout)
        self.setupSections(layout)
        self.show()

    def setupLogo(self, layout):
        """
        Sets up the logo display at the top of the application with dynamic sizing based on the screen resolution.
        This ensures the logo is appropriately scaled for different display sizes, enhancing visual appearance and usability.
        
        Args:
            layout (QLayout): The layout where the logo should be added, typically the main layout of the window.
        """
        # Create a QLabel to display the logo.
        self.logoLabel = QLabel()

        # Obtain the primary screen's resolution to determine appropriate scaling for the logo.
        screen = QApplication.primaryScreen()
        size = screen.size()
        # Choose the size of the pixmap based on the width of the screen.
        if size.width() <= 1550:  # Assuming 1550 is typical for smaller, laptop screens
            pixmap_size = 350
        else:
            pixmap_size = 580  # Assuming larger screens have at least this much width

        # Load the logo image, scale it according to the determined size, and maintain aspect ratio and smooth transformation.
        logoPixmap = QPixmap("./ui/FlowVis3D_logo_v2.jpg").scaled(pixmap_size, pixmap_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Set the scaled pixmap to the label.
        self.logoLabel.setPixmap(logoPixmap)
        # Align the logo in the center of the label, which improves layout and appearance.
        self.logoLabel.setAlignment(Qt.AlignCenter)
        # Add the logo label to the provided layout.
        layout.addWidget(self.logoLabel)

    def setupSections(self, layout):
        """
        Setup different collapsible sections of the application.
        """
        self.setupVisualizationSection(layout)
        self.setupRegistrationSection(layout)
        self.setupAdditionalRegistrationComponents()
        self.setupAdditionalSections(layout)

    def setupVisualizationSection(self, layout):
        """
        Sets up the visualization section in the application's user interface.
        This section is dedicated to loading and viewing point cloud data, specifically in the .ply file format.

        Args:
            layout (QVBoxLayout): The layout into which this section is to be integrated.
        """
        # Descriptive text that provides information on what the user can do in the "Load" section of the interface.
        info_text_load = (
            "Load Section:\n"
            "\u2022 Select the point cloud (.ply file format).\n"
            "\u2022 Display the 3D data."
        )

        # Create a collapsible section titled "Load" with the descriptive text and add it to the provided layout.
        self.visualizationSection = CollapsibleSection("Load", self, info_text=info_text_load)
        layout.addWidget(self.visualizationSection)

        # Create a button that allows users to select a point cloud file for loading.
        self.loadFileButton = QPushButton("Select Point Cloud")
        applyButtonStyle(self.loadFileButton)  # Apply predefined styling to the button.
        # Connect the button's click event to open a file dialog, specifying the directory to start in.
        self.loadFileButton.clicked.connect(lambda: self.openFileDialog(self.loadFileLineEdit, "\\\\srvnetapp00\\Technical\\Aerodynamics\\Development\\FlowViz"))
        self.visualizationSection.contentLayout().addWidget(self.loadFileButton)  # Add the button to the visualization section's layout.

        # Create a line edit for displaying the path of the loaded file, which is read-only to prevent manual editing.
        self.loadFileLineEdit = QLineEdit()
        self.loadFileLineEdit.setReadOnly(True)
        self.loadFileLineEdit.setPlaceholderText("Select or drag and drop .ply file")
        self.loadFileLineEdit.setProperty('role', 'source')  # Custom property to identify the role of the QLineEdit.
        self.loadFileLineEdit.setAcceptDrops(True)  # Allow drag-and-drop operations directly into the QLineEdit.
        applyLineEditStyle(self.loadFileLineEdit)  # Apply predefined styling to the line edit.
        self.visualizationSection.contentLayout().addWidget(self.loadFileLineEdit)  # Add the line edit to the layout.

        # Create a button to initiate the visualization of the loaded point cloud.
        self.visualizeButton = QPushButton("Visualize Point Cloud")
        applyButtonStyle(self.visualizeButton)  # Apply styling.
        self.visualizeButton.clicked.connect(self.startPointCloudVisualization)  # Connect the click event to the visualization function.
        self.visualizationSection.contentLayout().addWidget(self.visualizeButton)  # Add the button to the layout.

        # Create and setup a loading animation indicator, which will be shown while the point cloud is being loaded.
        self.loadingLabel_visualize, self.loadingMovie_visualize = createLoadingWheel(self.visualizationSection)  # This function returns a label and an animation object.

    def setupRegistrationSection(self, layout):
        """
        Sets up the registration section in the application's user interface.
        This section is dedicated to managing the registration of point cloud data. It includes functionality for selecting reference files,
        adjusting registration parameters, and handling registration operations.

        Args:
            layout (QVBoxLayout): The layout into which this section is to be integrated.
        """
        # Descriptive text that provides information on what the user can do in the "Registration" section of the interface.
        info_text_registration = (
            "Registration Section:\n"
            "\u2022 Select reference point cloud file. Two formats, .ply and .stl, are supported. The .ply format is the standard format. "
            "If the .stl format is chosen, it will be converted into the .ply format for the registration procedure, resulting in additional computational time.\n"
            "\u2022 Align point clouds using the registration algorithm (RANSAC Global Registration and ICP). Due to the randomic nature of the algorithm, it may be necessary to "
            "repeat the registration procedure a second time to obtain satisfying results.\n"
            "\u2022 Adjust registration settings such as voxel size and desired accuracy (optional).\n"
            "\u2022 Copy transformation matrix.\n"
            "\u2022 Display registered point cloud for inspection.\n"
            "\u2022 Save the registered data: save the registered point cloud ('_registered.ply'), the scaled registered point cloud ('_registered_paraview.ply') "
            "and, if available, the scaled registered mesh ('_registered_mesh_paraview.ply') in the input point cloud original folder. "
            "The scaled data is intended for paraview/sandbox, where one can compare FlowVis with CFD data.\n"
            "\u2022 Upload to sandbox (work in progress)."
        )

        # Create a collapsible section titled "Register" with the descriptive text and add it to the provided layout.
        self.registrationSection = CollapsibleSection("Register", self, info_text=info_text_registration)
        layout.addWidget(self.registrationSection)

        # Create a button that allows users to select a reference point cloud file for registration.
        self.loadRefFileButton = QPushButton("Select Reference")
        applyButtonStyle(self.loadRefFileButton)  # Apply predefined styling to the button.
        # Connect the button's click event to open a file dialog, specifying the directory to start in.
        self.loadRefFileButton.clicked.connect(lambda: self.openFileDialog(self.loadRefFileLineEdit, "\\\\srvnetapp00\\Technical\\Aerodynamics\\Development\\FlowViz"))
        self.registrationSection.contentLayout().addWidget(self.loadRefFileButton)  # Add the button to the registration section's layout.

        # Create a line edit for displaying the path of the loaded reference file, which is read-only to prevent manual editing.
        self.loadRefFileLineEdit = QLineEdit()
        self.loadRefFileLineEdit.setReadOnly(True)
        self.loadRefFileLineEdit.setPlaceholderText("Select or drag and drop .ply file")
        self.loadRefFileLineEdit.setProperty('role', 'reference')  # Custom property to identify the role of the QLineEdit.
        self.loadRefFileLineEdit.setAcceptDrops(True)  # Allow drag-and-drop operations directly into the QLineEdit.
        applyLineEditStyle(self.loadRefFileLineEdit)  # Apply predefined styling to the line edit.
        self.registrationSection.contentLayout().addWidget(self.loadRefFileLineEdit)  # Add the line edit to the layout.

        # Add a settings section for adjusting registration parameters like voxel size and accuracy.
        self.settingsSection = SettingsSection("Settings", self)
        self.registrationSection.contentLayout().addWidget(self.settingsSection)

        # Create a button to initiate the registration of the point cloud.
        self.executeRegistrationButton = QPushButton("Register Point Cloud")
        applyButtonStyle(self.executeRegistrationButton)  # Apply styling.
        self.executeRegistrationButton.clicked.connect(self.executeRegistration)  # Connect the click event to the registration function.
        self.registrationSection.contentLayout().addWidget(self.executeRegistrationButton)  # Add the button to the layout.

        # Create and setup a loading animation indicator, which will be shown while the registration process is running.
        self.loadingLabel_registration, self.loadingMovie_registration = createLoadingWheel(self.registrationSection)  # This function returns a label and an animation object.

        # Setup a logging area for registration operations.
        self.setupRegistrationLog()  # This method configures a log area to display registration process messages.

    def setupRegistrationLog(self):
        """
        Setup the log area for registration operations.
        """
        self.registrationLogLabel = QTextEdit()
        self.registrationLogLabel.setReadOnly(True)
        self.registrationLogLabel.setWordWrapMode(QTextOption.WrapAnywhere)
        self.registrationLogLabel.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.registrationLogLabel.setLayoutDirection(Qt.RightToLeft)
        applyTextAndScrollBarStyle(self.registrationLogLabel)
        self.registrationLogLabel.setFixedHeight(120)
        self.registrationSection.contentLayout().addWidget(self.registrationLogLabel)

    def setupAdditionalRegistrationComponents(self):
        """
        Configures additional interactive components within the registration section of the user interface.
        This method adds functionality to copy transformation matrices, visualize and save registered point clouds,
        and manage data upload operations with visual feedback for each process.
        """

        # Create a button that allows users to copy the transformation matrix to the clipboard.
        self.copyMatrixButton = QPushButton("Copy Transformation Matrix")
        applyButtonStyle(self.copyMatrixButton)  # Apply predefined styling to the button.
        self.copyMatrixButton.clicked.connect(self.copyTransformationToClipboard)  # Connect the button's click event to the copy functionality.
        self.registrationSection.contentLayout().addWidget(self.copyMatrixButton)  # Add the button to the registration section's layout.

        # Create a button to initiate the visualization of the registered point cloud.
        self.visualizeRegisteredButton = QPushButton("Visualize Registered Point Cloud")
        applyButtonStyle(self.visualizeRegisteredButton)  # Apply styling.
        self.visualizeRegisteredButton.clicked.connect(self.visualizeRegisteredPointCloud)  # Connect the click event to the visualization function.
        self.registrationSection.contentLayout().addWidget(self.visualizeRegisteredButton)  # Add the button to the layout.

        # Create a button to initiate the saving of the registered point cloud data.
        self.saveDataButton = QPushButton("Save Data")
        applyButtonStyle(self.saveDataButton)  # Apply styling.
        self.saveDataButton.clicked.connect(self.initiateSaveData)  # Connect the click event to the save data function.
        self.registrationSection.contentLayout().addWidget(self.saveDataButton)  # Add the button to the layout.

        # Create and setup a loading animation indicator for the saving operation, which will be shown while data is being saved.
        self.loadingLabel_savedata, self.loadingMovie_savedata = createLoadingWheel(self.registrationSection)  # This function returns a label and an animation object.

        # Setup a text edit widget for logging save operation messages.
        self.saveLogLabel = QTextEdit()
        self.saveLogLabel.setReadOnly(True)  # Make the log read-only.
        self.saveLogLabel.setWordWrapMode(QTextOption.WrapAnywhere)  # Enable word wrapping.
        self.saveLogLabel.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Add vertical scroll bar if needed.
        self.saveLogLabel.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Add horizontal scroll bar if needed.
        applyTextAndScrollBarStyle(self.saveLogLabel)  # Apply predefined styling.
        self.saveLogLabel.setFixedHeight(75)  # Set a fixed height for the log display.
        self.registrationSection.contentLayout().addWidget(self.saveLogLabel)  # Add the log display to the layout.

        # Create a button for uploading registered data to a sandbox environment.
        self.uploadSandboxButton = QPushButton("Upload to Sandbox")
        applyButtonStyle(self.uploadSandboxButton)  # Apply styling.
        self.uploadSandboxButton.clicked.connect(self.uploadtoSandbox)  # Connect the click event to the upload function.
        self.registrationSection.contentLayout().addWidget(self.uploadSandboxButton)  # Add the button to the layout.

        # Setup a text edit widget for logging upload operation messages.
        self.sandboxLogLabel = QTextEdit()
        self.sandboxLogLabel.setReadOnly(True)  # Make the log read-only.
        self.sandboxLogLabel.setWordWrapMode(QTextOption.WrapAnywhere)  # Enable word wrapping.
        self.sandboxLogLabel.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Add vertical scroll bar if needed.
        self.sandboxLogLabel.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Add horizontal scroll bar if needed.
        applyTextAndScrollBarStyle(self.sandboxLogLabel)  # Apply predefined styling.
        self.sandboxLogLabel.setFixedHeight(75)  # Set a fixed height for the log display.
        self.registrationSection.contentLayout().addWidget(self.sandboxLogLabel)  # Add the log display to the layout.

    def setupAdditionalSections(self, layout):
        """
        Setup additional sections such as Pre-process, Segment, and Raster (currently place-holders).
        """
        info_text_preprocessing = (
            "Segmentation Section (work in progress):\n"
            "\u2022 Downsample the input registered point cloud.\n"
            "\u2022 Estimate the point cloud normals."
        )
        info_text_segmentation = (
            "Segmentation Section (work in progress):\n"
            "\u2022 Divide the point cloud into coarse surfaces using K-means.\n"
            "\u2022 Segment the point cloud into coherent surfaces using DBSCAN."
        )
        info_text_raster = (
            "Raster Section (work in progress):\n"
            "\u2022 Orient each point cloud segment using PCA.\n"
            "\u2022 Project each PCA-rotated segmented onto an optimal 2D plane and obtain a 2D representation of the segment."
        )
        preprocessSection = CollapsibleSection("Pre-process", self, info_text=info_text_preprocessing)
        segmentSection = CollapsibleSection("Segment", self, info_text=info_text_segmentation)
        rasterSection = CollapsibleSection("Raster", self, info_text=info_text_raster)
        layout.addWidget(preprocessSection)
        layout.addWidget(segmentSection)
        layout.addWidget(rasterSection)

    def openFileDialog(self, lineEdit, initial_dir):
        """
        Opens a file dialog allowing the user to select a point cloud file from the filesystem.
        This method updates the provided QLineEdit with the selected file path and adjusts the application state based on the selection.

        Args:
            lineEdit (QLineEdit): The QLineEdit widget to update with the selected file path.
            initial_dir (str): The initial directory to open in the file dialog.
        """
        # Create a file dialog object with a specific title and parent (self refers to the MainApp class).
        dialog = QFileDialog(self, "Select Point Cloud File")
        dialog.setDirectory(initial_dir)  # Set the initial directory of the file dialog.
        dialog.setNameFilter("Point Cloud Files (*.ply *.stl);;All Files (*)")  # Set the file type filters.
        dialog.setFileMode(QFileDialog.ExistingFile)  # Allow selection of only existing files.
        dialog.setViewMode(QFileDialog.Detail)  # Set the view mode to show detailed information about files.
        dialog.setModal(True)  # Make the dialog modal, blocking interactions with other windows until it is closed.
        dialog.resize(800, 600)  # Set the initial size of the dialog.
        dialog.setMinimumSize(800, 600)  # Set the minimum size of the dialog to prevent resizing to a smaller window.
        dialog.setWindowState(Qt.WindowNoState)  # Ensure the window state is normal (not minimized, maximized, or fullscreen).

        # Execute the dialog and check if the user accepted (clicked OK).
        if dialog.exec() == QFileDialog.Accepted:
            selectedFiles = dialog.selectedFiles()  # Retrieve the list of selected files.
            if selectedFiles:  # Check if there is at least one selected file.
                file_name = selectedFiles[0]  # Get the first file from the list.
                lineEdit.setText(file_name)  # Update the QLineEdit widget with the selected file path.

                # Specific behavior if the line edit is for loading point clouds, enables visualization button.
                if lineEdit is self.loadFileLineEdit:
                    self.file_path_pcd = file_name  # Update the stored file path for the point cloud.
                    self.visualizeButton.setDisabled(False)  # Enable the visualization button when a file is selected.

                # Specific behavior if the line edit is for loading reference files in registration, enables registration button.
                if lineEdit is self.loadRefFileLineEdit:
                    self.visualizeRegisteredButton.setDisabled(False)  # Enable the registration button when a reference is selected.

    def dragEnterEvent(self, event: QDragEnterEvent):
        """
        Handles the event when a drag operation enters the application window. This method is triggered
        whenever a drag operation involves the application's GUI.

        Args:
            event (QDragEnterEvent): Contains information about the drag event, including the data being dragged.
        """
        # Check if the drag data includes URLs, which typically represent file paths when dragging files from the file explorer.
        if event.mimeData().hasUrls():
            # If there are URLs, accept the drag action. This allows the dropEvent method to be called if the user drops the files.
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """
        Handles the drop event when a user releases a dragged item (e.g., file) over the application window.
        This function specifically updates text fields based on the dropped file's path, depending on where it was dropped.

        Args:
            event (QDropEvent): Contains information about the drop action, including the data and the position of the drop.
        """
        # Extract the URLs from the drop event's MIME data. URLs represent the paths of the files being dropped.
        urls = event.mimeData().urls()
        if urls:  # Check if there are any URLs
            path = urls[0].toLocalFile()  # Convert the first URL to a local file path.
            cursor_position = event.pos()  # Get the position of the cursor at the time of the drop.

            # Determine which widget is directly beneath the cursor at the drop position.
            child = self.childAt(cursor_position)
            if child:
                # If the widget directly under the cursor isn't a QLineEdit, traverse up the widget hierarchy to find the nearest parent QLineEdit.
                while not isinstance(child, QLineEdit) and child is not None:
                    child = child.parent()
                if child and isinstance(child, QLineEdit):  # Check if a QLineEdit was found.
                    # Retrieve a custom property ('role') from the QLineEdit to determine its intended role (source or reference).
                    role = child.property('role')
                    if role == 'source':
                        self.loadFileLineEdit.setText(path)  # Update the 'source' line edit with the path of the dropped file.
                    elif role == 'reference':
                        self.loadRefFileLineEdit.setText(path)  # Update the 'reference' line edit with the path.

    def getFilePath(self):
        """
        Retrieves the current text from the fileLineEdit widget, which typically holds the path to a selected file.

        Returns:
            str: The text content of the fileLineEdit, representing the file path.
        """
        # Return the text content of the fileLineEdit, which is expected to be a file path.
        return self.fileLineEdit.text()

    def setLoadFilePath(self, file_path):
        """
        Sets the text of the filePathLineEdit widget to the specified file path. This is typically used
        to update the UI with a new path after a file has been selected or changed.

        Args:
            file_path (str): The file path to set in the filePathLineEdit widget.
        """
        # Update the filePathLineEdit with the new file path, reflecting changes or new selections in the UI.
        self.filePathLineEdit.setText(file_path)

    def setReferenceFilePath(self, file_path):
        """
        Sets the text of the referenceFilePathLineEdit widget to the specified reference file path. This method is used
        to update the path displayed in the referenceFilePathLineEdit, typically after selecting a new reference file.

        Args:
            file_path (str): The reference file path to set in the referenceFilePathLineEdit widget.
        """
        # Update the referenceFilePathLineEdit with the new reference file path, reflecting the user's selection or changes.
        self.referenceFilePathLineEdit.setText(file_path)


    def startPointCloudVisualization(self):
        """
        Initiates the process of loading and visualizing a point cloud based on the file path provided in the loadFileLineEdit.
        This method checks if the file path has changed or if the point cloud is not already loaded before starting the loading process.
        """
        # Retrieve the file path from the loadFileLineEdit widget.
        file_path = self.loadFileLineEdit.text()
        
        # Check if a file path is not specified.
        if not file_path:
            QMessageBox.warning(self, "Warning", "No file selected. Please load a point cloud first.")
            return

        # Check if the point cloud needs to be reloaded (path change or not previously loaded).
        if self.cached_pcd_path != file_path or self.cached_pcd is None:
            # Start the loading animation to indicate to the user that processing is underway.
            startLoadingAnimation(self.loadingLabel_visualize, self.loadingMovie_visualize, self.visualizationSection.scrollArea)

            # Initialize a worker and thread for loading the point cloud from the specified file path.
            self.loadPointCloudWorker = LoadPointCloudWorker(file_path=file_path)
            self.loadPointCloudThread = QThread()
            self.loadPointCloudWorker.moveToThread(self.loadPointCloudThread)

            # Connect the finished signals of the worker to the appropriate slots.
            self.loadPointCloudWorker.finished.connect(self.handleLoadPointCloudComplete)
            self.loadPointCloudWorker.finished.connect(self.loadPointCloudThread.quit)
            self.loadPointCloudWorker.finished.connect(self.loadPointCloudWorker.deleteLater)
            self.loadPointCloudThread.finished.connect(self.loadPointCloudThread.deleteLater)

            # Start the thread, which triggers the worker's run method.
            self.loadPointCloudThread.started.connect(self.loadPointCloudWorker.run)
            self.loadPointCloudThread.start()
        else:
            # If the point cloud is already loaded and the path hasn't changed, directly visualize the cached point cloud.
            self.visualizePointCloud(self.cached_pcd, "")

    def handleLoadPointCloudComplete(self, pcd, error_message):
        """
        Handles the completion of the point cloud loading process. This method is called when the LoadPointCloudWorker
        has finished processing, and it manages both successful and failed loads.

        Args:
            pcd: The loaded point cloud object or None if loading failed.
            error_message (str): A message describing the error if one occurred during loading.
        """
        # Stop the loading animation regardless of success or error.
        stopLoadingAnimation(self.loadingLabel_visualize, self.loadingMovie_visualize, self.visualizationSection.scrollArea)

        # Handle errors during point cloud loading.
        if error_message:
            QMessageBox.critical(self, "Visualization Error", error_message)
            return

        # If a point cloud was successfully loaded, proceed with visualization.
        if pcd:
            # Cache the successfully loaded point cloud and its path.
            self.cached_pcd = pcd
            self.cached_pcd_path = self.loadFileLineEdit.text()

            # Attempt to visualize the loaded point cloud using Open3D.
            try:
                o3d.visualization.draw_geometries([pcd], window_name='FlowVis3D')
            except Exception as e:
                QMessageBox.critical(self, "Visualization Error", f"Failed to visualize the point cloud: {str(e)}")
        else:
            # Alert the user if no point cloud was loaded.
            QMessageBox.warning(self, "Load Warning", "No point cloud was loaded.")

    def visualizePointCloud(self, pcd, error_message):
        """
        Visualizes a given point cloud if no error occurred during the loading or processing stages.
        This method attempts to render the point cloud using Open3D visualization tools.

        Args:
            pcd: The point cloud data to visualize. This is typically an object compatible with Open3D visualization.
            error_message (str): An error message indicating if something went wrong before attempting visualization.
        """
        # Check if there is an error message and display it if present.
        if error_message:
            QMessageBox.critical(self, "Visualization Error", error_message)
            # Uncomment below to stop loading animation if used elsewhere in the application context.
            # stopLoadingAnimation(self.loadingLabel_visualize, self.loadingMovie_visualize, self.visualizationSection.scrollArea)
            return

        # If the point cloud data is not None, attempt to visualize it.
        if pcd is not None:
            try:
                # Use Open3D to draw geometries; this function will open a visualization window.
                o3d.visualization.draw_geometries([pcd], window_name='FlowVis3D')
            except Exception as e:
                # If visualization fails, catch the exception and show an error message.
                QMessageBox.critical(self, "Visualization Error", f"Failed to visualize the point cloud: {str(e)}")

        # If applicable, stop any ongoing animations that indicate loading or processing.
        # stopLoadingAnimation(self.loadingLabel_visualize, self.loadingMovie_visualize, self.visualizationSection.scrollArea)

    def handleLoadError(self, error_message):
        """
        Handles errors that may occur during the loading of point cloud data.
        This method displays an error message and stops any visual indications of ongoing processes, such as animations.

        Args:
            error_message (str): A message describing the error that occurred during the loading process.
        """
        # Display a critical error message using a message box.
        QMessageBox.critical(self, "Load Error", f"Failed to load the point cloud: {error_message}")
        # Stop any loading animations to indicate that the process has ended, possibly due to an error.
        stopLoadingAnimation(self.loadingLabel_visualize, self.loadingMovie_visualize)

    def executeRegistration(self):
        """
        Initiates the point cloud registration process by setting up and starting a new worker thread.
        This method also handles user input validation and manages UI updates related to the registration process.
        """
        # Retrieve the registration settings from the settings section.
        voxel_sizes, desired_fitness_ransac, desired_fitness_icp = self.settingsSection.getSettings() 

        # Clear any previous messages in the registration log to prepare for new output.
        self.registrationLogLabel.clear()

        # Ensure that any previously running registration threads are properly terminated before starting a new one.
        if self.registrationThread is not None:
            if self.registrationThread.isRunning():
                self.registrationThread.quit()
                self.registrationThread.wait()

        # Start the loading animation in the UI to indicate that registration is processing.
        startLoadingAnimation(self.loadingLabel_registration, self.loadingMovie_registration, self.registrationSection.scrollArea)

        # Fetch the file paths for the source and reference point clouds from the respective QLineEdit widgets.
        sourceFilePath = self.loadFileLineEdit.text()
        referenceFilePath = self.loadRefFileLineEdit.text()

        # Check if both necessary files are selected, and show a warning if either is missing.
        if not sourceFilePath or not referenceFilePath:
            missing_files = []
            if not sourceFilePath:
                missing_files.append("point cloud")
            if not referenceFilePath:
                missing_files.append("reference")
            missing = " and ".join(missing_files)
            QMessageBox.warning(self, "Warning", f"Please load {missing}.")
            stopLoadingAnimation(self.loadingLabel_registration, self.loadingMovie_registration, self.registrationSection.scrollArea)
            return

        # Initialize the registration worker with the file paths and registration settings.
        self.registrationWorker = RegistrationWorker(sourceFilePath, referenceFilePath, self, voxel_sizes, desired_fitness_ransac, desired_fitness_icp=desired_fitness_icp)
        self.registrationThread = QThread()  # Create a new thread for registration operations.
        self.registrationWorker.moveToThread(self.registrationThread)  # Move the worker to the new thread.

        # Connect the signals from the worker to appropriate slots to handle completion and errors.
        self.registrationWorker.finished.connect(self.handleRegistrationComplete)
        self.registrationWorker.error.connect(self.handleRegistrationError)

        # Print to console for debugging purposes indicating that signal connections are established.
        # print("Connecting signals")

        # Start the worker thread and handle its lifecycle.
        self.registrationThread.started.connect(self.registrationWorker.run)
        self.registrationThread.finished.connect(self.registrationThread.deleteLater)
        self.registrationThread.start()  # Begin the registration process.

    def handleRegistrationComplete(self, pcd_registered, transformation, log_text):
        """
        Handles the successful completion of the registration process. Updates the UI and caches the results.

        Args:
            pcd_registered: The registered point cloud returned by the registration worker.
            transformation: The transformation matrix resulting from the registration.
            log_text (str): Text to log in the UI, typically containing details about the registration process.
        """
        # Print to console for debugging purposes to indicate the slot was triggered and to show received data.
        # print("Registration complete slot triggered.")
        # print("Received data:", pcd_registered)

        # Stop any ongoing loading animations in the UI, signaling that the registration has completed.
        stopLoadingAnimation(self.loadingLabel_registration, self.loadingMovie_registration, self.registrationSection.scrollArea)

        # Update the registration log in the UI with the provided log text.
        self.registrationLogLabel.setText(log_text)

        # Cache the successfully registered point cloud and transformation matrix for later use.
        self.registered_pcd = pcd_registered
        self.transformation = transformation

        # Print to console for debugging purposes to indicate data storage state post-registration.
        # print("Data stored in MainApp immediately after registration:", self.registered_pcd)

    def handleRegistrationError(self, error_message):
        """
        Handles errors that occur during the registration process. This method updates the UI to reflect the error and stops any loading animations.

        Args:
            error_message (str): A message describing what went wrong during registration.
        """
        # Stop any ongoing loading animations to indicate that the registration process has halted due to an error.
        stopLoadingAnimation(self.loadingLabel_registration, self.loadingMovie_registration, self.registrationSection.scrollArea)

        # Update the registration log in the UI with the error message to inform the user what went wrong.
        self.registrationLogLabel.setText(f"Registration failed: {error_message}")

    def copyTransformationToClipboard(self):
        """
        Copies the transformation matrix, if available, to the system clipboard. This allows the user to paste it elsewhere.

        """
        # Access the system clipboard through the QGuiApplication.
        clipboard = QGuiApplication.clipboard()
        # Convert the transformation matrix to string format if it exists and copy it to the clipboard.
        transformation_text = str(self.transformation)  # Ensure this contains the transformation matrix text.
        clipboard.setText(transformation_text)
        # Print to console for debugging purposes to confirm that the matrix has been copied.
        # print("Transformation matrix copied to clipboard.")

    def visualizeRegisteredPointCloud(self):
        """
        Visualizes the registered point cloud if it's available and if the reference file necessary for comparison is loaded.
        """
        # Print to console for debugging purposes about the visualization attempt.
        # print("Visualizing registered point cloud. Current data:", self.registered_pcd)

        # Check if there's a registered point cloud available to visualize.
        if not self.registered_pcd:
            QMessageBox.warning(self, "Visualization Error", "No registered point cloud available. Please perform registration first.")
            return

        # Check if the reference file path is loaded.
        current_path = self.loadRefFileLineEdit.text().strip()
        if not current_path:
            QMessageBox.warning(self, "Visualization Error", "No reference file loaded. Please load a reference file.")
            return

        # Determine if the geometry needs to be reloaded due to a change in the reference path or if it's not loaded.
        if current_path != self.cached_reference_path or self.reference_geometry is None:
            # Reset the reference geometry and update the cached path.
            self.reference_geometry = None
            self.cached_reference_path = current_path

            # Clean up any existing visualization infrastructure before setting up a new load operation.
            self.tearDownVisualizationInfrastructure()

            # Set up a new worker to load the reference geometry based on the current path.
            self.setupLoadReferenceWorker(current_path)
        else:
            # If the geometry is already loaded and the path hasn't changed, proceed to visualize it directly.
            self.visualizeData(self.reference_geometry)

    def setupLoadReferenceWorker(self, path):
        """
        Sets up a worker and a thread to load reference geometry from the specified path. This method ensures
        that any previous workers and threads are properly cleaned up before initiating a new load operation.

        Args:
            path (str): The file path from which the reference geometry is to be loaded.
        """
        # First, clean up any existing visualization infrastructure to avoid conflicts.
        self.tearDownVisualizationInfrastructure()

        # Create a new worker for loading the reference geometry.
        self.LoadReferenceWorker = LoadReferenceWorker(path)
        self.visualizationThread = QThread()  # Create a new thread.

        # Move the newly created worker to the new thread.
        self.LoadReferenceWorker.moveToThread(self.visualizationThread)

        # Connect various signals of the worker to appropriate slots.
        # These include data loaded successfully, errors, and cleanup signals.
        self.LoadReferenceWorker.dataLoaded.connect(self.cacheAndVisualizeData)
        self.LoadReferenceWorker.error.connect(self.handleVisualizationError)
        self.LoadReferenceWorker.finished.connect(self.LoadReferenceWorker.deleteLater)
        self.LoadReferenceWorker.finished.connect(self.visualizationThread.quit)

        # Start the thread which in turn starts the worker's run method.
        self.visualizationThread.started.connect(self.LoadReferenceWorker.run)
        self.visualizationThread.finished.connect(self.visualizationThread.deleteLater)
        self.visualizationThread.start()

    def tearDownVisualizationInfrastructure(self):
        """
        Cleans up any existing visualization workers and threads to ensure a clean state before starting new operations.
        """
        # Check if there is an existing LoadReferenceWorker and if it is active.
        if self.LoadReferenceWorker:
            if self.LoadReferenceWorker.active:
                # Attempt to disconnect any connected signals to prevent memory leaks and ensure proper cleanup.
                try:
                    self.LoadReferenceWorker.dataLoaded.disconnect()
                    self.LoadReferenceWorker.error.disconnect()
                    self.LoadReferenceWorker.finished.disconnect()
                except RuntimeError as e:
                    print(f"Error while disconnecting signals: {e}")
            # Safely delete the worker.
            self.LoadReferenceWorker.deleteLater()
            self.LoadReferenceWorker = None

        # Check if there is an existing thread for visualization and ensure it is properly terminated.
        if self.visualizationThread:
            if self.visualizationThread.isRunning():
                self.visualizationThread.quit()
                self.visualizationThread.wait()
            self.visualizationThread.deleteLater()
            self.visualizationThread = None

    def cacheAndVisualizeData(self, reference_geom):
        """
        Caches the loaded reference geometry and triggers its visualization. This method is typically called
        when the reference geometry has been successfully loaded by a worker.

        Args:
            reference_geom: The reference geometry that has been loaded and is now ready for visualization.
        """
        # Cache the loaded reference geometry for future use or re-use without reloading.
        self.reference_geometry = reference_geom
        # Call the method to visualize the reference geometry now stored in cache.
        self.visualizeData(reference_geom)

    def visualizeData(self, reference_geom):
        """
        Visualizes the loaded reference geometry alongside the registered point cloud using Open3D visualization tools.
        This method attempts to set up a visualization environment, add the geometries to it, and display them.

        Args:
            reference_geom: The geometry to be visualized along with the registered point cloud.
        """
        try:
            # Initialize a new Open3D Visualizer object.
            self.vis = o3d.visualization.Visualizer()
            # Create a visualization window with a specified name.
            self.vis.create_window(window_name='FlowVis3D')
            # Add the registered point cloud to the visualizer.
            self.vis.add_geometry(self.registered_pcd)  # Registered point cloud
            # Also add the reference geometry that has just been loaded or updated.
            self.vis.add_geometry(reference_geom)       # Loaded reference geometry
            # Run the visualizer to display the geometries until the window is closed.
            self.vis.run()
            # Once the visualization is done or the window is closed, properly destroy the window to free resources.
            self.vis.destroy_window()
        except Exception as e:
            # If any errors occur during the setup or visualization process, handle them appropriately.
            self.handleVisualizationError(str(e))

    def handleVisualizationError(self, error_message):
        """
        Handles any errors that may occur during the visualization process by logging the error details.
        This method can be expanded to include more sophisticated error handling strategies such as user notifications.

        Args:
            error_message (str): A message describing the error encountered during visualization.
        """
        # Print the error message to the console. This could be replaced or supplemented by more user-visible error reporting mechanisms.
        print(f"Visualization Error: {error_message}")

    def initiateSaveData(self):
        """
        Initiates the save data process by first verifying that a file has been selected and checking for existing files.
        This method ensures that any previous saved data isn't unintentionally overwritten without user consent.
        """
        # Clear any previous messages in the save log.
        self.saveLogLabel.clear()

        # Check if a file has been selected to save.
        if not self.file_path_pcd:
            QMessageBox.warning(self, "Warning", "No file selected. Please select a file first.")
            return

        # Construct file paths for both raw and scaled versions of the registered point cloud.
        output_file_raw = self.file_path_pcd.replace('.ply', '_registered.ply')
        output_file_scaled = self.file_path_pcd.replace('.ply', '_registered_paraview.ply')

        # Check if either of the constructed file paths already exists.
        if os.path.exists(output_file_raw) or os.path.exists(output_file_scaled):
            # Prompt the user to confirm overwriting existing files.
            reply = QMessageBox.question(self, 'Confirm Overwrite',
                                        "Files already exist. Do you want to overwrite them?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            # If the user chooses not to overwrite, cancel the save operation.
            if reply == QMessageBox.No:
                QMessageBox.information(self, "Save Cancelled", "Save operation cancelled by user.")
                return

        # Proceed with the save process if the files do not exist or if overwrite is confirmed.
        self.startSaveProcess()

    def startSaveProcess(self):
        """
        Starts the actual save process by setting up a SaveWorker within a new thread.
        This method manages the lifecycle of the save operation, including starting and cleaning up the thread.
        """
        # Check if a save operation is already in progress and warn the user if so.
        if self.saveThread and self.saveThread.isRunning():
            QMessageBox.warning(self, "Warning", "A save operation is already in progress. Please wait for it to complete.")
            return

        # Instantiate the SaveWorker with the necessary data for saving.
        self.saveWorker = SaveWorker(self, self.file_path_pcd, self.registered_pcd, self.transformation, save_mesh=True)
        # Connect the worker's signals to appropriate slots for handling completion, errors, and log messages.
        self.saveWorker.finished.connect(self.cleanupSaveProcess)
        self.saveWorker.error.connect(self.handleSaveError)
        self.saveWorker.log_message.connect(self.updateSaveLog)

        # Create and configure a new thread for the save operation.
        self.saveThread = QThread()
        self.saveWorker.moveToThread(self.saveThread)
        self.saveThread.started.connect(self.saveWorker.run)
        self.saveThread.finished.connect(self.saveThread.deleteLater)

        # Start the save thread and initiate the loading animation to indicate saving is in progress.
        self.saveThread.start()
        startLoadingAnimation(self.loadingLabel_savedata, self.loadingMovie_savedata, self.registrationSection.scrollArea)

    def handleSaveError(self, error):
        """
        Handles any errors that occur during the save process by displaying an error message to the user and initiating cleanup.

        Args:
            error (str): The error message describing what went wrong during the save operation.
        """
        # Display a critical error message using a message box to alert the user that an error has occurred.
        QMessageBox.critical(self, "Save Error", f"An error occurred: {error}")
        # After displaying the error message, call the cleanup process to properly shut down the worker and thread.
        self.cleanupSaveProcess()  # Ensures that resources are freed and the application state is correctly reset.

    def cleanupSaveProcess(self):
        """
        Cleans up resources and UI components used during the save process. This method is called after a save operation
        completes, either successfully or due to an error, to ensure the application returns to a stable state.
        """
        # Stop any animations in the UI that indicate ongoing save operations, signaling completion to the user.
        stopLoadingAnimation(self.loadingLabel_savedata, self.loadingMovie_savedata, self.registrationSection.scrollArea)

        # Check if there is an active save worker and ensure it is properly deleted to free resources.
        if self.saveWorker:
            self.saveWorker.deleteLater()  # Safely delete the worker object.
            self.saveWorker = None         # Remove the reference to the worker.

        # Check if there is an active save thread and ensure it is properly terminated.
        if self.saveThread:
            self.saveThread.quit()        # Request the thread to quit.
            self.saveThread.wait()        # Wait for the thread to finish.
            self.saveThread.deleteLater() # Safely delete the thread object.
            self.saveThread = None

        # Uncomment below if you wish to provide user feedback when save operations are definitively completed.
        # QMessageBox.information(self, "Save Complete", "Data has been successfully saved.")

    def updateSaveLog(self, message):
        """
        Updates the save log text edit widget with messages received from the SaveWorker.
        This function is typically used to display status updates and results from the save process.

        Args:
            message (str): The message to be added to the save log.
        """
        # Append the provided message to the save log QTextEdit widget.
        # This allows users to see the progress and details of the save operation in real-time.
        self.saveLogLabel.append(message)

    def appendLog(self, message):
        """
        Appends a given message to the save log. This method can be used for general logging
        outside of the specific save operation context.

        Args:
            message (str): The message to append to the log.
        """
        # Append the provided message to the save log QTextEdit widget.
        # This method is a general utility for logging various types of messages.
        self.saveLogLabel.append(message)  # Append message to the QTextEdit

    def uploadtoSandbox(self):
        """
        Initiates the upload of registered data to a sandbox environment. This function is
        a placeholder and indicates that the functionality is currently under development.

        """
        # Clear any existing messages in the sandbox log label to prepare for new information.
        self.sandboxLogLabel.clear()
        # Set the text of the sandbox log label to indicate that the upload functionality is still being developed.
        log_text_sandbox = 'Upload to Sandbox currently under development'
        self.sandboxLogLabel.setText(log_text_sandbox)

# Run the script as the main program.
if __name__ == "__main__":
    # Initialize the QApplication object
    app = QApplication([])  # The list can hold command line arguments passed to the application.

    # Set a dark theme for the entire application by modifying the style sheet of the application.
    # This changes the background color to black and the text color to white for all QWidget objects.
    app.setStyleSheet("""
        QWidget {
            background-color: black;
            color: white;
        }
    """)

    # Set a custom font for the entire application.
    nunitoFont = QFont("Nunito", 10)  
    app.setFont(nunitoFont)  # Apply the font to the application, affecting all text in the UI.

    # Create an instance of the MainApp class. 
    window = MainApp()

    # Execute the application's main loop
    app.exec()
