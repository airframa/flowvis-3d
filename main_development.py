from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLineEdit, QFileDialog, QLabel, QTextEdit,  QMessageBox
from PySide6.QtGui import QFont, QPixmap, QDragEnterEvent, QDropEvent, QTextOption, QGuiApplication, QMovie
from PySide6.QtCore import Qt, QTimer, QObject, Signal, Slot, QThread
from IPython.display import clear_output
from components.point_cloud_utils import load_pcd
from components.point_cloud_registration import PointCloudRegistration
import open3d as o3d
import os

print('Yah mon, starting up...')  # Should print immediately


class CollapsibleSection(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # The header contains the title and the toggle button
        self.header = QHBoxLayout()

        # The title label
        self.titleLabel = QLabel(title)
        self.titleLabel.setStyleSheet("font-weight: bold;")  # Bold font for the title
        self.header.addWidget(self.titleLabel)

        # The toggle button
        self.toggleButton = QPushButton("+")
        self.toggleButton.setFixedSize(30, 30)  # Slightly larger button size
        self.toggleButton.setStyleSheet("font-size: 18px; font-weight: bold;")  # Larger text and bold
        self.toggleButton.setCheckable(True)
        self.toggleButton.setChecked(False)
        self.toggleButton.clicked.connect(self.onToggle)
        self.header.addWidget(self.toggleButton)

        # Ensure that the title and button are aligned to the left, with no expanding space in between
        self.header.addStretch(1)

        self.layout.addLayout(self.header)

        # The content widget is hidden initially and shown when the toggle button is clicked
        self.contentWidget = QWidget()
        self.contentWidget.setLayout(QVBoxLayout())
        self.contentWidget.setVisible(False)
        self.layout.addWidget(self.contentWidget)

    def onToggle(self):
        # Toggle the content visibility
        isVisible = self.toggleButton.isChecked()
        self.contentWidget.setVisible(isVisible)
        self.toggleButton.setText("-" if isVisible else "+")

        # Directly request an update and re-layout
        self.updateGeometry()  # Suggest to Qt that it should recalculate geometries
        self.parentWidget().layout().activate()  # Force the parent layout to reevaluate its size constraints

        # Optionally, force the top-level window to adjust to the new layout
        topLevelWindow = self.window()
        if topLevelWindow:
            topLevelWindow.adjustSize()  # Adjust the size of the top-level window to fit its contents

    def contentLayout(self):
        # Provides access to the content area's layout
        return self.contentWidget.layout()
    
def applyButtonStyle(button):
    button.setStyleSheet("""
        QPushButton {
            background-color: red;
            color: white;
            font-weight: bold;
            border-radius: 12px; /* Rounded corners */
            padding: 5px 15px;
            margin: 5px;
        }
    """)

def applyLineEditStyle(lineEdit):
    lineEdit.setStyleSheet("""
        QLineEdit {
            background-color: #f0f0f0;  /* Light grey background */
            color: #333;  /* Dark text color */
            border-radius: 12px;  /* Rounded corners to match the buttons */
            padding: 5px 15px;  /* Padding to match the buttons */
            font-size: 14px;  /* Adjust font size as needed */
            margin: 5px;  /* Margin to ensure consistency with button spacing */
        }
        QLineEdit:read-only {
            background-color: #e0e0e0;  /* Slightly darker background for read-only mode */
        }
    """)

def applyTextAndScrollBarStyle(widget):
    baseStyle = """
    QTextEdit {
        color: #333;
        padding: 5px 15px;
        font-size: 14px;
        margin: 5px;
        background-color: #f0f0f0;  /* Slightly grayish white */
        border: none;
        border-radius: 12px;
    }
    QTextEdit:read-only {
        background-color: #e0e0e0;  /* A bit darker for read-only mode */
    }
    QTextEdit QAbstractScrollArea::viewport {
        background: #e0e0e0;  /* Ensure viewport background matches the QTextEdit */
        border-radius: 12px;
    }
    """
    
    scrollbarStyle = """
    QTextEdit QScrollBar:vertical {
        border: none;
        background: #e0e0e0;  /* Ensure scrollbar track matches the QTextEdit background */
        width: 10px;
        margin: 0px 0 0px 0;
        border-radius: 0px;
    }
    QTextEdit QScrollBar::handle:vertical {
        background-color: #5b5b5b;
        min-height: 30px;
        border-radius: 5px;
    }
    QTextEdit QScrollBar::handle:vertical:hover {
        background-color: #5b5b5b;
    }
    QTextEdit QScrollBar::add-line:vertical, QTextEdit QScrollBar::sub-line:vertical {
        border: none;
        background: none;
        height: 0px;
    }
    QTextEdit QScrollBar::add-page:vertical, QTextEdit QScrollBar::sub-page:vertical {
        background: none;
    }
    """
    
    # Combine both styles and apply to the widget
    widget.setStyleSheet(baseStyle + scrollbarStyle)


class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setWindowTitle("FlowVis3D")
        self.setAcceptDrops(True)
        self.setupUI()
        self.registrationThread = None
        self.registrationWorker = None
        self.visualizationThread = None
        self.visualizationWorker = None
        self.reference_geometry = None
        self.cached_reference_path = None
        self.registered_pcd = None  # Assuming this holds your registered point cloud
        self.transformation = None  # Assuming this holds the transformation used in registration
        self.workerIsActive = False  # Flag to track if the worker is active

    def setupUI(self):
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)

        # Load and display the logo at the top
        self.logoLabel = QLabel()
        # self.logoPixmap = QPixmap("./ui/FlowVis3D_logo_v2.jpg").scaled(580, 580, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logoPixmap = QPixmap("./ui/FlowVis3D_logo_v2.jpg").scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logoLabel.setPixmap(self.logoPixmap)
        self.logoLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.logoLabel)

        # Visualization Section with CollapsibleSection
        visualizationSection = CollapsibleSection("Load", self)
        layout.addWidget(visualizationSection)

        # File chooser setup for loading point cloud
        self.loadFileButton = QPushButton("Select Point Cloud")
        applyButtonStyle(self.loadFileButton)
        visualizationSection.contentLayout().addWidget(self.loadFileButton)
        # Connect button click to openFileDialog method, passing the line edit as an argument
        self.loadFileButton.clicked.connect(lambda: self.openFileDialog(self.loadFileLineEdit, "\\\\srvnetapp00\\Technical\\Aerodynamics\\Development\\FlowViz"))

        # Setup for line edit with rounded edges and placeholder text
        self.loadFileLineEdit = QLineEdit()
        self.loadFileLineEdit.setReadOnly(True)
        self.loadFileLineEdit.setPlaceholderText("Select or drag and drop .ply file")
        self.loadFileLineEdit.setProperty('role', 'source')
        self.loadFileLineEdit.setAcceptDrops(True)
        applyLineEditStyle(self.loadFileLineEdit)
        visualizationSection.contentLayout().addWidget(self.loadFileLineEdit)

        # Show 3D Point Cloud Button within the visualization section
        self.visualizeButton = QPushButton("Visualize Point Cloud")
        applyButtonStyle(self.visualizeButton)
        self.visualizeButton.clicked.connect(self.visualizePointCloud)
        visualizationSection.contentLayout().addWidget(self.visualizeButton)

        # Register Section with CollapsibleSection
        registrationSection = CollapsibleSection("Register", self)
        layout.addWidget(registrationSection)

        # File chooser setup for loading point cloud
        self.loadRefFileButton = QPushButton("Select Reference")
        applyButtonStyle(self.loadRefFileButton)
        registrationSection.contentLayout().addWidget(self.loadRefFileButton)
        # Connect button click to openFileDialog method, passing the line edit as an argument
        self.loadRefFileButton.clicked.connect(lambda: self.openFileDialog(self.loadRefFileLineEdit, "\\\\srvnetapp00\\Technical\\Aerodynamics\\Development\\FlowViz"))

        # Allow for drag and drop
        self.loadRefFileLineEdit = QLineEdit()
        self.loadRefFileLineEdit.setReadOnly(True)
        self.loadRefFileLineEdit.setPlaceholderText("Select or drag and drop .ply file")
        self.loadRefFileLineEdit.setProperty('role', 'reference')
        self.loadRefFileLineEdit.setAcceptDrops(True)
        applyLineEditStyle(self.loadRefFileLineEdit)
        registrationSection.contentLayout().addWidget(self.loadRefFileLineEdit)

        # Execute Registration Button within the registration section
        self.executeRegistrationButton = QPushButton("Register Point Cloud")
        applyButtonStyle(self.executeRegistrationButton)
        self.executeRegistrationButton.clicked.connect(self.executeRegistration)
        registrationSection.contentLayout().addWidget(self.executeRegistrationButton)

        # Spinning wheel setup
        wheelLayout = QHBoxLayout()
        wheelLayout.addStretch()  # Add stretch to push everything to the center

        self.loadingLabel = QLabel()
        self.loadingMovie = QMovie("./ui/spinning_wheel_v2_60.gif")
        self.loadingLabel.setMovie(self.loadingMovie)
        self.loadingLabel.setAlignment(Qt.AlignCenter)
        self.loadingLabel.setFixedSize(25, 25)
        self.loadingMovie.setScaledSize(self.loadingLabel.size())
        self.loadingLabel.hide()

        wheelLayout.addWidget(self.loadingLabel)
        wheelLayout.addStretch()  # Add stretch to ensure the label is centered

        # Insert the wheel layout into the registration section's layout
        registrationSection.contentLayout().addLayout(wheelLayout)

       # Registration logs display with combined text area and scrollbar style
        self.registrationLogLabel = QTextEdit()
        self.registrationLogLabel.setReadOnly(True)
        self.registrationLogLabel.setWordWrapMode(QTextOption.WrapAnywhere)
        self.registrationLogLabel.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show vertical scrollbar only when needed
        self.registrationLogLabel.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show horizontal scrollbar only when needed
        applyTextAndScrollBarStyle(self.registrationLogLabel)  # Apply combined styles
        self.registrationLogLabel.setFixedHeight(120)
        registrationSection.contentLayout().addWidget(self.registrationLogLabel)

        # Button to copy the transformation matrix to the clipboard
        self.copyMatrixButton = QPushButton("Copy Transformation Matrix")
        # Style for the copy matrix button
        applyButtonStyle(self.copyMatrixButton)
        self.copyMatrixButton.clicked.connect(self.copyTransformationToClipboard)
        registrationSection.contentLayout().addWidget(self.copyMatrixButton)

        # Visualization Registered Button
        self.visualizeRegisteredButton = QPushButton("Visualize Registered Point Cloud")
        applyButtonStyle(self.visualizeRegisteredButton)
        self.visualizeRegisteredButton.clicked.connect(self.visualizeRegisteredPointCloud)
        registrationSection.contentLayout().addWidget(self.visualizeRegisteredButton)

        # Save Data
        self.saveDataButton = QPushButton("Save Data")
        applyButtonStyle(self.saveDataButton)
        self.saveDataButton.clicked.connect(self.initiateSaveData)
        registrationSection.contentLayout().addWidget(self.saveDataButton)

        # Spinning wheel setup
        wheelLayout2 = QHBoxLayout()
        wheelLayout2.addStretch()  # Add stretch to push everything to the center

        self.loadingLabel2 = QLabel()
        self.loadingMovie2 = QMovie("./ui/spinning_wheel_v2_60.gif")
        self.loadingLabel2.setMovie(self.loadingMovie2)
        self.loadingLabel2.setAlignment(Qt.AlignCenter)
        self.loadingLabel2.setFixedSize(25, 25)
        self.loadingMovie2.setScaledSize(self.loadingLabel2.size())
        self.loadingLabel2.hide()

        wheelLayout2.addWidget(self.loadingLabel2)
        wheelLayout2.addStretch()  # Add stretch to ensure the label is centered

        # Insert the wheel layout into the registration section's layout
        registrationSection.contentLayout().addLayout(wheelLayout2)

        # Registration logs display with combined text area and scrollbar style
        self.saveLogLabel = QTextEdit()
        self.saveLogLabel.setReadOnly(True)
        self.saveLogLabel.setWordWrapMode(QTextOption.WrapAnywhere)
        self.saveLogLabel.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show vertical scrollbar only when needed
        self.saveLogLabel.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show horizontal scrollbar only when needed
        applyTextAndScrollBarStyle(self.saveLogLabel)  # Apply combined styles
        self.saveLogLabel.setFixedHeight(75)
        registrationSection.contentLayout().addWidget(self.saveLogLabel)

        # Spinning wheel setup for the visualization process
        # # Spinning wheel setup
        # wheelLayout2 = QHBoxLayout()
        # wheelLayout2.addStretch()  # Add stretch to push everything to the center

        # self.visualizationLoadingLabel = QLabel()
        # self.visualizationLoadingMovie = QMovie("./ui/spinning_wheel_v2_60.gif")
        # self.visualizationLoadingLabel.setMovie(self.visualizationLoadingMovie)
        # self.visualizationLoadingLabel.setAlignment(Qt.AlignCenter)
        # self.visualizationLoadingLabel.setFixedSize(25, 25)
        # self.visualizationLoadingMovie.setScaledSize(self.visualizationLoadingLabel.size())
        # self.visualizationLoadingLabel.hide()  # Initially hidden
        # # registrationSection.contentLayout().addWidget(self.visualizationLoadingLabel)

        # wheelLayout2.addWidget(self.visualizationLoadingLabel)
        # wheelLayout2.addStretch()  # Add stretch to ensure the label is centered

        # # Insert the wheel layout into the registration section's layout
        # registrationSection.contentLayout().addLayout(wheelLayout2)

        # Preprocessing Section (placeholder)
        preprocessSection = CollapsibleSection("Pre-process", self)
        layout.addWidget(preprocessSection)
        # Populate preprocessSection.contentLayout() as needed in the future

        # Segmentation Section (placeholder)
        segmentSection = CollapsibleSection("Segment", self)
        layout.addWidget(segmentSection)
        # Populate segmentSection.contentLayout() as needed in the future

        # Segmentation Section (placeholder)
        rasterSection = CollapsibleSection("Raster", self)
        layout.addWidget(rasterSection)
        # Populate rasterSection.contentLayout() as needed in the future

        self.show()

    def openFileDialog(self, lineEdit, initial_dir):
        dialog = QFileDialog(self, "Select Point Cloud File")
        dialog.setDirectory(initial_dir)
        dialog.setNameFilter("Point Cloud Files (*.ply *.stl);;All Files (*)")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setViewMode(QFileDialog.Detail)
        dialog.setModal(True)
        dialog.resize(800, 600)
        dialog.setMinimumSize(800, 600)
        dialog.setWindowState(Qt.WindowNoState)

        if dialog.exec() == QFileDialog.Accepted:
            selectedFiles = dialog.selectedFiles()
            if selectedFiles:
                file_name = selectedFiles[0]
                lineEdit.setText(file_name)
                if lineEdit is self.loadFileLineEdit:
                    self.visualizeButton.setDisabled(False)  # Enable visualization button when file is selected
                if lineEdit is self.loadRefFileLineEdit:
                    self.visualizeRegisteredButton.setDisabled(False)  # Enable registration button when reference is selected

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            cursor_position = event.pos()
            # Check which QLineEdit is under the cursor when the drop happens
            child = self.childAt(cursor_position)
            if child:
                # Find the nearest parent QLineEdit if the direct child isn't one
                while not isinstance(child, QLineEdit) and child is not None:
                    child = child.parent()
                if child and isinstance(child, QLineEdit):
                    role = child.property('role')
                    if role == 'source':
                        self.loadFileLineEdit.setText(path)
                    elif role == 'reference':
                        self.loadRefFileLineEdit.setText(path)

    def getFilePath(self):
        return self.fileLineEdit.text()

    def setLoadFilePath(self, file_path):
        self.filePathLineEdit.setText(file_path)

    def setReferenceFilePath(self, file_path):
        self.referenceFilePathLineEdit.setText(file_path)

    # Define the method to load and visualize the point cloud using Open3D.
    def visualizePointCloud(self):
        # Use the text from the line edit directly
        file_path = self.loadFileLineEdit.text()
        if file_path:
            try:
                pcd = load_pcd(file_path)  # Load the point cloud data.
                o3d.visualization.draw_geometries([pcd])  # Visualize the point cloud.
            except Exception as e:
                print(f"Failed to load or visualize the point cloud: {e}")
        else:
            print("No file selected.")

    def executeRegistration(self):
        # Ensure previous worker and thread are properly cleaned up
        if self.registrationThread is not None:
            if self.registrationThread.isRunning():
                self.registrationThread.quit()
                self.registrationThread.wait()

        self.startLoadingAnimation()
        sourceFilePath = self.loadFileLineEdit.text()
        referenceFilePath = self.loadRefFileLineEdit.text()

        # Create new worker and thread
        self.registrationWorker = RegistrationWorker(sourceFilePath, referenceFilePath, self)
        self.registrationThread = QThread()
        self.registrationWorker.moveToThread(self.registrationThread)

        # Connect the finished signal to the handler
        print("Connecting signals")
        self.registrationWorker.finished.connect(self.handleRegistrationComplete)
        self.registrationWorker.error.connect(self.handleRegistrationError)

        # Setup and start the thread
        self.registrationThread.started.connect(self.registrationWorker.run)
        self.registrationThread.finished.connect(self.registrationThread.deleteLater)
        self.registrationThread.start()

    def handleRegistrationComplete(self, pcd_registered, transformation, log_text):
        print("Registration complete slot triggered.")
        print("Received data:", pcd_registered)
        self.stopLoadingAnimation()
        self.registrationLogLabel.setText(log_text)
        self.registered_pcd = pcd_registered
        self.transformation = transformation
        print("Data stored in MainApp immediately after registration:", self.registered_pcd)


    def handleRegistrationError(self, error_message):
        self.stopLoadingAnimation()
        self.registrationLogLabel.setText(f"Registration failed: {error_message}")
        # Additional error handling

    # def handleRegistrationComplete(self, pcd_registered, transformation, log_text):
    #     self.stopLoadingAnimation()
    #     self.transformationMatrixText = transformation
    #     self.pcd_registered = pcd_registered  # Store the registered point cloud
    #     self.registrationLogLabel.setText(f"{log_text}")

    def handleRegistrationError(self, error_message):
        self.stopLoadingAnimation()
        self.registrationLogLabel.setText(f"Registration failed: {error_message}")

    def startLoadingAnimation(self):
        self.loadingLabel.show()
        self.loadingMovie.start()

    def stopLoadingAnimation(self):
        self.loadingMovie.stop()
        self.loadingLabel.hide()

    def copyTransformationToClipboard(self):
        clipboard = QGuiApplication.clipboard()
        transformation_text = self.transformationMatrixText  # Ensure this contains the transformation matrix text
        clipboard.setText(transformation_text)
        print("Transformation matrix copied to clipboard.")

    def visualizeRegisteredPointCloud(self):
        print("Visualizing registered point cloud. Current data:", self.registered_pcd)
        current_path = self.loadRefFileLineEdit.text()
        if current_path != self.cached_reference_path or self.reference_geometry is None:
            self.reference_geometry = None
            self.cached_reference_path = current_path
            
            # Make sure any ongoing visualization is properly cleaned up before starting a new one
            self.tearDownVisualizationInfrastructure()

            # Set up a new visualization worker
            self.setupVisualizationWorker(current_path)
        else:
            # If the geometry is already loaded and the path hasn't changed, visualize directly
            self.visualizeData(self.reference_geometry)

    def setupVisualizationWorker(self, path):
        # Clean up any existing worker and thread first
        self.tearDownVisualizationInfrastructure()

        self.visualizationWorker = VisualizationWorker(path)
        self.visualizationThread = QThread()

        # Move the worker to the thread
        self.visualizationWorker.moveToThread(self.visualizationThread)

        # Connect signals
        self.visualizationWorker.dataLoaded.connect(self.cacheAndVisualizeData)
        self.visualizationWorker.error.connect(self.handleVisualizationError)
        self.visualizationWorker.finished.connect(self.visualizationWorker.deleteLater)
        self.visualizationWorker.finished.connect(self.visualizationThread.quit)

        # Start the thread
        self.visualizationThread.started.connect(self.visualizationWorker.run)
        self.visualizationThread.finished.connect(self.visualizationThread.deleteLater)

        self.visualizationThread.start()

    def tearDownVisualizationInfrastructure(self):
        if self.visualizationWorker:
            if self.visualizationWorker.active:
                # Disconnect signals
                try:
                    self.visualizationWorker.dataLoaded.disconnect()
                    self.visualizationWorker.error.disconnect()
                    self.visualizationWorker.finished.disconnect()
                except RuntimeError as e:
                    print(f"Error while disconnecting signals: {e}")
            # Delete the worker safely
            self.visualizationWorker.deleteLater()
            self.visualizationWorker = None

        if self.visualizationThread:
            if self.visualizationThread.isRunning():
                self.visualizationThread.quit()
                self.visualizationThread.wait()
            self.visualizationThread.deleteLater()
            self.visualizationThread = None

    def cacheAndVisualizeData(self, reference_geom):
        self.reference_geometry = reference_geom
        self.visualizeData(reference_geom)

    def visualizeData(self, reference_geom):
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            print(self.registered_pcd)
            vis.add_geometry(self.registered_pcd)  # Registered point cloud
            vis.add_geometry(reference_geom)       # Loaded reference geometry
            vis.run()
            vis.destroy_window()
        except Exception as e:
            self.handleVisualizationError(str(e))

    def handleVisualizationError(self, error_message):
        print(f"Visualization Error: {error_message}")

    def initiateSaveData(self):
        print("Attempting to save data. Current registered pcd:", self.registered_pcd)
        file_path = self.loadFileLineEdit.text()
        print(self.registered_pcd)
        if not file_path:
            QMessageBox.warning(self, "Warning", "No file selected. Please load a point cloud first.")
            return
        if not self.registered_pcd:
            QMessageBox.warning(self, "Warning", "No registered point cloud available. Perform registration first.")
            return

        # Start the save operation in a new thread
        self.saveThread = QThread()
        self.saveWorker = SaveWorker(file_path, self.registered_pcd, self.transformation, save_mesh=True)
        self.saveWorker.moveToThread(self.saveThread)
        
        # Connect signals to slots
        self.saveWorker.finished.connect(self.saveThread.quit)
        self.saveWorker.finished.connect(self.saveWorker.deleteLater)
        self.saveWorker.finished.connect(self.stopSaveLoadingAnimation)  # Stop animation when finished
        self.saveWorker.error.connect(lambda error: QMessageBox.critical(self, "Error", f"An error occurred: {error}"))
        self.saveWorker.error.connect(self.stopSaveLoadingAnimation)  # Stop animation on error
        self.saveWorker.log_message.connect(self.appendLog)  # Connect log message signal

        # Start spinning animation and thread
        self.startSaveLoadingAnimation()  # Start animation when save is initiated
        self.saveThread.started.connect(self.saveWorker.run)
        self.saveThread.finished.connect(self.saveThread.deleteLater)
        
        self.saveThread.start()

    def appendLog(self, message):
        self.saveLogLabel.append(message)  # Append message to the QTextEdit

    def startSaveLoadingAnimation(self):
        self.loadingLabel2.show()
        self.loadingMovie2.start()

    def stopSaveLoadingAnimation(self):
        self.loadingMovie2.stop()
        self.loadingLabel2.hide()


    # Optionally, reset any progress indicators
    # self.progressBar.reset()  # If you have a progress bar, reset it
    # self.statusLabel.setText("Ready to start a new session")  # Reset status label if exists


    # def removeVisualizationLoadingAnimation(self):
    #     print("Removing the loading animation...")  # Debug print
    #     if self.visualizationLoadingLabel.isVisible():
    #         self.visualizationLoadingMovie.stop()
    #         self.visualizationLoadingLabel.hide()  # Directly remove the spinning wheel

    # def startVisualizationLoadingAnimation(self):
    #     print("Starting the loading animation...")  # Debug print
    #     self.visualizationLoadingLabel.show()
    #     self.visualizationLoadingMovie.start()

    # def stopVisualizationLoadingAnimation(self):
    #     print("Stopping the loading animation...")  # Debug print
    #     self.visualizationLoadingMovie.stop()
    #     self.visualizationLoadingLabel.hide()  # Hide the QLabel

    # def handleVisualizationError(self, error_message):
    #     print("Visualization Error:", error_message)  # Debug print
    #     self.registrationLogLabel.setText(f"Visualization failed: {error_message}")
    #     self.stopVisualizationLoadingAnimation()


class RegistrationWorker(QObject):
    finished = Signal(object, str, str)  # Emitted with the registered point cloud, transformation matrix, and log message
    error = Signal(str)                 # Emitted with error message
    requestStop = Signal()
    active = False                      # Attribute to manage active state

    def __init__(self, sourcePath, referencePath, main_app):
        super(RegistrationWorker, self).__init__()
        self.sourcePath = sourcePath
        self.referencePath = referencePath
        self.main_app = main_app
        self.active = True

    def stop(self):
        if self.active:
            self.active = False  # Set active to False to stop any ongoing processes safely

    @Slot()
    def run(self):
        if not self.active:
            return
        try:
            source_pcd = load_pcd(self.sourcePath)
            reference_pcd = load_pcd(self.referencePath)
            registration = PointCloudRegistration(source=source_pcd, target=reference_pcd)

            # Assume 'register' method returns a tuple of (registered_pcd, transformation_matrix, log_text)
            pcd_registered, transformation, log_text = registration.register(
                desired_fitness_ransac=0.85,
                desired_fitness_icp=[0.65, 0.75, 0.85]
            )

            if self.active:
                # Emit the successful registration results
                self.finished.emit(pcd_registered, str(transformation), log_text)
                print("Emitting finished signal with:", pcd_registered, transformation, log_text)

        except Exception as e:
            if self.active:
                self.error.emit(str(e))
        finally:
            self.active = False  # Ensure the worker is marked as inactive

    def deleteLater(self):
        if self.active:
            self.requestStop.emit()  # Ensure stop is requested before deletion
        super(RegistrationWorker, self).deleteLater()


class VisualizationWorker(QObject):
    dataLoaded = Signal(object)  # Emitted when data is loaded successfully
    error = Signal(str)          # Emitted on error
    finished = Signal()          # Emitted when processing is complete
    active = False               # Manage the active state

    def __init__(self, reference_path):
        super(VisualizationWorker, self).__init__()
        self.reference_path = reference_path
        self.active = True

    @Slot()
    def run(self):
        if not self.active:
            return
        try:
            # Assuming this loads a mesh or point cloud
            reference_geom = o3d.io.read_triangle_mesh(self.reference_path)
            reference_geom.compute_vertex_normals()
            if self.active:
                self.dataLoaded.emit(reference_geom)  # Emit loaded geometry
        except Exception as e:
            if self.active:
                self.error.emit(str(e))
        finally:
            self.active = False

    def deleteLater(self):
        if self.active:
            self.active = False  # Ensure no more operations are executed
        super(VisualizationWorker, self).deleteLater()

class SaveWorker(QObject):
    finished = Signal()
    error = Signal(str)
    log_message = Signal(str)  # Signal to send log messages

    def __init__(self, file_path, registered_pcd, transformation, save_mesh=False):
        super(SaveWorker, self).__init__()
        self.file_path = file_path
        self.registered_pcd = registered_pcd
        self.transformation = transformation
        self.save_mesh = save_mesh

    @Slot()
    def run(self):
        try:
            if self.registered_pcd is not None:
                output_file_raw = self.file_path.replace('.ply', '_registered.ply')
                output_file_scaled = self.file_path.replace('.ply', '_registered_paraview.ply')
                o3d.io.write_point_cloud(output_file_raw, self.registered_pcd)

                scale = 1 / 600
                self.registered_pcd.scale(scale, center=(0, 0, 0))
                o3d.io.write_point_cloud(output_file_scaled, self.registered_pcd)

                saved_files = [
                    "original registered point cloud",
                    "CFD-scaled registered point cloud"
                ]

                if self.save_mesh:
                    try:
                        mesh = o3d.io.read_triangle_mesh(self.file_path)
                        if len(mesh.triangles) > 0:
                            mesh.transform(self.transformation)
                            mesh.scale(scale, center=(0, 0, 0))
                            output_mesh_file = self.file_path.replace('.ply', '_registered_mesh_paraview.ply')
                            o3d.io.write_triangle_mesh(output_mesh_file, mesh)
                            saved_files.append("CFD-scaled registered mesh")
                        else:
                            summary_message = "No mesh data available for the current .ply file\n"
                    except Exception as e:
                        print(f"Failed to process mesh data: {str(e)}")
                        summary_message = "Failed to read mesh data\n"
                
                # Use the directory path
                directory_path = os.path.dirname(self.file_path)
                summary_message += f"Registered data ({', '.join(saved_files)}) saved in the directory: {directory_path}"
                self.log_message.emit(summary_message)
            else:
                self.log_message.emit("No registered point cloud available. Perform registration first.")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


if __name__ == "__main__":
    app = QApplication([])

    # Set the dark theme for the application
    app.setStyleSheet("""
        QWidget {
            background-color: black;
            color: white;
        }
    """)

    # Set a custom font for the entire application
    nunitoFont = QFont("Nunito", 10)  # Adjust the size as needed
    app.setFont(nunitoFont)

    window = MainApp()
    app.exec()