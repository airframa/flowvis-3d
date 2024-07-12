# class CollapsibleSection(QWidget):
#     def __init__(self, title, parent=None, height_threshold=350):
#         super().__init__(parent)
#         self.height_threshold = height_threshold
#         self.layout = QVBoxLayout(self)
        
#         # Setting main background to black
#         self.setStyleSheet("background-color: black;")

#         # Header setup
#         self.header = QHBoxLayout()
#         self.titleLabel = QLabel(title)
#         self.titleLabel.setStyleSheet("font-weight: bold; color: white;")
#         self.header.addWidget(self.titleLabel)

#         self.toggleButton = QPushButton("+")
#         self.toggleButton.setFixedSize(30, 30)
#         self.toggleButton.setStyleSheet("font-size: 18px; font-weight: bold; color: white; background-color: black;")
#         self.toggleButton.setCheckable(True)
#         self.toggleButton.setChecked(False)
#         self.toggleButton.clicked.connect(self.onToggle)
#         self.header.addWidget(self.toggleButton)

#         self.header.addStretch(1)
#         self.layout.addLayout(self.header)

#         # Content setup with scroll area
#         self.scrollArea = QScrollArea(self)
#         self.scrollArea.setWidgetResizable(True)
#         self.scrollArea.setVisible(False)
#         self.scrollArea.setLayoutDirection(Qt.RightToLeft)  # Set the layout direction to RightToLeft
#         self.scrollArea.setStyleSheet("""
#             QScrollArea {
#                 border: none;
#                 background: black;
#             }
#             QScrollArea QScrollBar:vertical {
#                 border: none;
#                 background: black;
#                 width: 10px;
#                 margin: 0px 0 0px 0;
#                 border-radius: 0px;
#             }
#             QScrollArea QScrollBar::handle:vertical {
#                 background-color: #5b5b5b;
#                 min-height: 30px;
#                 border-radius: 5px;
#             }
#             QScrollArea QScrollBar::handle:vertical:hover {
#                 background-color: #5b5b5b;
#             }
#             QScrollArea QScrollBar::add-line:vertical, QScrollArea QScrollBar::sub-line:vertical {
#                 border: none;
#                 background: none;
#                 height: 0px;
#             }
#             QScrollArea QScrollBar::add-page:vertical, QScrollArea QScrollBar::sub-page:vertical {
#                 background: none;
#             }
#             QScrollArea QWidget#viewport {
#                 background: black;
#             }
#         """)  # Custom scroll bar style and ensuring viewport background is black

#         self.contentWidget = QWidget()
#         self.contentWidget.setObjectName("viewport")  # This ensures the background style applies correctly
#         self.contentWidget.setLayout(QVBoxLayout())
#         self.scrollArea.setWidget(self.contentWidget)

#         self.layout.addWidget(self.scrollArea)

#     def onToggle(self):
#         isVisible = self.toggleButton.isChecked()
#         self.scrollArea.setVisible(isVisible)
#         self.toggleButton.setText("-" if isVisible else "+")

#         if isVisible:
#             self.adjustScrollArea()
#         self.updateGeometry()
#         if self.parentWidget():
#             self.parentWidget().layout().activate()

#         topLevelWindow = self.window()
#         if topLevelWindow:
#             topLevelWindow.adjustSize()

#     def adjustScrollArea(self):
#         # Adjusts the presence of scrollbars depending on content height
#         contentHeight = self.contentWidget.sizeHint().height()
#         if contentHeight > self.height_threshold:
#             self.scrollArea.setFixedSize(QSize(self.width(), self.height_threshold))
#             self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
#         else:
#             self.scrollArea.setFixedSize(QSize(self.width(), contentHeight))
#             self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

#     def contentLayout(self):
#         return self.contentWidget.layout()
    
# class SettingsSection(CollapsibleSection):
#     def __init__(self, title, parent=None):
#         # Set a very large height threshold to avoid scroll bar appearance
#         super().__init__(title, parent, height_threshold=10000)
#         self.initUI()

#     def initUI(self):
#         # Clear any existing widgets from the header
#         while self.header.count():
#             item = self.header.takeAt(0)
#             if item.widget():
#                 item.widget().deleteLater()

#         # Setup the titleLabel
#         self.titleLabel = QLabel("Settings")
#         self.titleLabel.setStyleSheet("font-weight: bold; color: white;")

#         # Adding the toggle button
#         self.toggleButton = QPushButton("+")
#         self.toggleButton.setFixedSize(30, 30)
#         self.toggleButton.setStyleSheet("font-size: 18px; font-weight: bold; color: white; background-color: black;")
#         self.toggleButton.setCheckable(True)
#         self.toggleButton.setChecked(False)
#         self.toggleButton.clicked.connect(self.onToggle)

#         # Add widgets to the header: order is reversed for a sub-section
#         self.header.addStretch(1)
#         self.header.addWidget(self.toggleButton)
#         self.header.addWidget(self.titleLabel)
       
#         # Initialize the scroll area and content widget
#         self.scrollArea.setWidgetResizable(True)
#         self.scrollArea.setVisible(False)
#         self.contentWidget = QWidget()
#         self.contentWidget.setLayout(QVBoxLayout())
#         self.scrollArea.setWidget(self.contentWidget)

#         # Example content widgets
#         self.voxelSizesEdit = QLineEdit("20.0, 25.0, 30.0, 35.0")
#         self.contentLayout().addWidget(QLabel("Voxel Sizes:"))
#         self.contentLayout().addWidget(self.voxelSizesEdit)
#         applyLineEditStyle(self.voxelSizesEdit)

#         self.fitnessRansacEdit = QLineEdit("0.85")
#         self.contentLayout().addWidget(QLabel("Desired Fitness RANSAC:"))
#         self.contentLayout().addWidget(self.fitnessRansacEdit)
#         applyLineEditStyle(self.fitnessRansacEdit)

#         self.fitnessIcpEdit = QLineEdit("0.65, 0.75, 0.85")
#         self.contentLayout().addWidget(QLabel("Desired Fitness ICP:"))
#         self.contentLayout().addWidget(self.fitnessIcpEdit)
#         applyLineEditStyle(self.fitnessIcpEdit)

#     def onToggle(self):
#         # Simply toggle the visibility based on the toggle button's state
#         isVisible = self.toggleButton.isChecked()
#         self.scrollArea.setVisible(isVisible)
#         self.toggleButton.setText("-" if isVisible else "+")

#         # Manually set visibility for all child widgets in the content layout
#         for i in range(self.contentWidget.layout().count()):
#             widget = self.contentWidget.layout().itemAt(i).widget()
#             if widget:
#                 widget.setVisible(isVisible)

#     def adjustScrollArea(self):
#         # Adjusts the presence of scrollbars depending on content height
#         contentHeight = self.contentWidget.sizeHint().height()
#         if contentHeight > self.height_threshold:
#             self.scrollArea.setFixedSize(QSize(self.width(), self.height_threshold))
#             self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
#         else:
#             self.scrollArea.setFixedSize(QSize(self.width(), contentHeight))
#             self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

#     def getSettings(self):
#         # Parse inputs from QLineEdit, converting them to appropriate types
#         voxel_sizes = list(map(float, self.voxelSizesEdit.text().split(',')))
#         desired_fitness_ransac = float(self.fitnessRansacEdit.text())
#         desired_fitness_icp = list(map(float, self.fitnessIcpEdit.text().split(',')))
#         return voxel_sizes, desired_fitness_ransac, desired_fitness_icp


# def applyButtonStyle(button):
#     button.setStyleSheet("""
#         QPushButton {
#             background-color: red;
#             color: white;
#             font-weight: bold;
#             border-radius: 12px; /* Rounded corners */
#             padding: 5px 15px;
#             margin: 5px;
#         }
#     """)

# def applyLineEditStyle(lineEdit):
#     lineEdit.setStyleSheet("""
#         QLineEdit {
#             background-color: #f0f0f0;  /* Light grey background */
#             color: #333;  /* Dark text color */
#             border-radius: 12px;  /* Rounded corners to match the buttons */
#             padding: 5px 15px;  /* Padding to match the buttons */
#             font-size: 14px;  /* Adjust font size as needed */
#             margin: 5px;  /* Margin to ensure consistency with button spacing */
#         }
#         QLineEdit:read-only {
#             background-color: #e0e0e0;  /* Slightly darker background for read-only mode */
#         }
#     """)

# def applyTextAndScrollBarStyle(widget):
#     baseStyle = """
#     QTextEdit {
#         color: #333;
#         padding: 5px 15px;
#         font-size: 14px;
#         margin: 5px;
#         background-color: #f0f0f0;  /* Slightly grayish white */
#         border: none;
#         border-radius: 12px;
#     }
#     QTextEdit:read-only {
#         background-color: #e0e0e0;  /* A bit darker for read-only mode */
#     }
#     QTextEdit QAbstractScrollArea::viewport {
#         background: #e0e0e0;  /* Ensure viewport background matches the QTextEdit */
#         border-radius: 12px;
#     }
#     """
    
#     scrollbarStyle = """
#     QTextEdit QScrollBar:vertical {
#         border: none;
#         background: #e0e0e0;  /* Ensure scrollbar track matches the QTextEdit background */
#         width: 10px;
#         margin: 0px 0 0px 0;
#         border-radius: 0px;
#     }
#     QTextEdit QScrollBar::handle:vertical {
#         background-color: #5b5b5b;
#         min-height: 30px;
#         border-radius: 5px;
#     }
#     QTextEdit QScrollBar::handle:vertical:hover {
#         background-color: #5b5b5b;
#     }
#     QTextEdit QScrollBar::add-line:vertical, QTextEdit QScrollBar::sub-line:vertical {
#         border: none;
#         background: none;
#         height: 0px;
#     }
#     QTextEdit QScrollBar::add-page:vertical, QTextEdit QScrollBar::sub-page:vertical {
#         background: none;
#     }
#     """
    
#     # Combine both styles and apply to the widget
#     widget.setStyleSheet(baseStyle + scrollbarStyle)

# def createLoadingWheel(parent_section):
#         wheelLayout = QHBoxLayout()
#         wheelLayout.addStretch()

#         loadingLabel = QLabel()
#         loadingMovie = QMovie("./ui/spinning_wheel_v2_60.gif")
#         loadingLabel.setMovie(loadingMovie)
#         loadingLabel.setAlignment(Qt.AlignCenter)
#         loadingLabel.setFixedSize(25, 25)
#         loadingMovie.setScaledSize(loadingLabel.size())
#         loadingLabel.hide()

#         wheelLayout.addWidget(loadingLabel)
#         wheelLayout.addStretch()
#         parent_section.contentLayout().addLayout(wheelLayout)

#         return loadingLabel, loadingMovie

# def startLoadingAnimation(label, movie, scrollArea):
#     label.show()
#     movie.start()
#     # Increase the height of the scroll area to accommodate the loading label
#     current_size = scrollArea.size()
#     updated_height = current_size.height() + 20  # Add an extra 25 pixels for the animation
#     scrollArea.setFixedSize(current_size.width(), updated_height)

# def stopLoadingAnimation(label, movie, scrollArea):
#     movie.stop()
#     label.hide()
#     # Restore the original height by subtracting the added 25 pixels
#     current_size = scrollArea.size()
#     updated_height = current_size.height() - 20  # Subtract the extra 25 pixels
#     scrollArea.setFixedSize(current_size.width(), updated_height)

# class LoadPointCloudWorker(QObject):
#     finished = Signal(object, str)  # Signal for point cloud and error message

#     def __init__(self, file_path=None, parent=None):
#         super().__init__(parent)
#         self.file_path = file_path

#     def run(self):
#         # Simulate loading a point cloud
#         try:
#             pcd = load_pcd(self.file_path)  # Assuming load_pcd is your function to load the point cloud
#             self.finished.emit(pcd, "")  # Emitting with no error
#         except Exception as e:
#             self.finished.emit(None, str(e))  # Emitting None and error message

# class RegistrationWorker(QObject):
#     finished = Signal(object, object, str)  # Emitted with the registered point cloud, transformation matrix, and log message
#     error = Signal(str)                 # Emitted with error message
#     requestStop = Signal()
#     active = False                      # Attribute to manage active state

#     def __init__(self, sourcePath, referencePath, main_app, voxel_sizes, desired_fitness_ransac, desired_fitness_icp):
#         super(RegistrationWorker, self).__init__()
#         self.sourcePath = sourcePath
#         self.referencePath = referencePath
#         self.main_app = main_app
#         self.voxel_sizes = voxel_sizes
#         self.desired_fitness_ransac = desired_fitness_ransac
#         self.desired_fitness_icp = desired_fitness_icp
#         self.active = True

#     def stop(self):
#         if self.active:
#             self.active = False  # Set active to False to stop any ongoing processes safely

#     @Slot()
#     def run(self):
#         if not self.active:
#             return
#         try:
#             source_pcd = load_pcd(self.sourcePath)
#             reference_pcd = load_pcd(self.referencePath)
#             registration = PointCloudRegistration(source=source_pcd, target=reference_pcd)

#             # Assume 'register' method returns a tuple of (registered_pcd, transformation_matrix, log_text)
#             pcd_registered, transformation, log_text = registration.register(
#                 voxel_sizes=self.voxel_sizes, 
#                 desired_fitness_ransac=self.desired_fitness_ransac,
#                 desired_fitness_icp=self.desired_fitness_icp
#             )

#             if self.active:
#                 # Emit the successful registration results
#                 # self.finished.emit(pcd_registered, str(transformation), log_text)
#                 self.finished.emit(pcd_registered, transformation, log_text)
#                 print("Emitting finished signal with:", pcd_registered, transformation, log_text)

#         except Exception as e:
#             if self.active:
#                 self.error.emit(str(e))
#         finally:
#             self.active = False  # Ensure the worker is marked as inactive

#     def deleteLater(self):
#         if self.active:
#             self.requestStop.emit()  # Ensure stop is requested before deletion
#         super(RegistrationWorker, self).deleteLater()


# class LoadReferenceWorker(QObject):
#     dataLoaded = Signal(object)  # Emitted when data is loaded successfully
#     error = Signal(str)          # Emitted on error
#     finished = Signal()          # Emitted when processing is complete
#     active = False               # Manage the active state

#     def __init__(self, reference_path):
#         super(LoadReferenceWorker, self).__init__()
#         self.reference_path = reference_path
#         self.active = True

#     @Slot()
#     def run(self):
#         if not self.active:
#             return
#         try:
#             # Assuming this loads a mesh or point cloud
#             reference_geom = o3d.io.read_triangle_mesh(self.reference_path)
#             reference_geom.compute_vertex_normals()
#             if self.active:
#                 self.dataLoaded.emit(reference_geom)  # Emit loaded geometry
#         except Exception as e:
#             if self.active:
#                 self.error.emit(str(e))
#         finally:
#             self.active = False

#     def deleteLater(self):
#         if self.active:
#             self.active = False  # Ensure no more operations are executed
#         super(LoadReferenceWorker, self).deleteLater()

# class SaveWorker(QObject):
#     finished = Signal()
#     error = Signal(str)
#     log_message = Signal(str)  # Signal to send log messages

#     def __init__(self, main_app, file_path, registered_pcd, transformation, save_mesh=False):
#         super(SaveWorker, self).__init__()
#         self.main_app = main_app
#         self.file_path = file_path
#         self.registered_pcd = registered_pcd
#         self.transformation = transformation
#         self.save_mesh = save_mesh

#     @Slot()
#     def run(self):
#         try:
#             if self.registered_pcd is not None:
#                 # Store registered point cloud
#                 pcd_copy = copy.deepcopy(self.registered_pcd)
#                 output_file_raw = self.file_path.replace('.ply', '_registered.ply')
#                 output_file_scaled = self.file_path.replace('.ply', '_registered_paraview.ply')
#                 o3d.io.write_point_cloud(output_file_raw, pcd_copy)
#                 # Store registered point cloud, sclaed
#                 scale = 1 / 600
#                 registered_pcd_scaled = pcd_copy.scale(scale, center=(0, 0, 0))
#                 o3d.io.write_point_cloud(output_file_scaled, registered_pcd_scaled)

#                 saved_files = [
#                     "original registered point cloud",
#                     "CFD-scaled registered point cloud"
#                 ]

#                 if self.save_mesh:
#                     try:
#                         mesh = o3d.io.read_triangle_mesh(self.file_path)
#                         if len(mesh.triangles) > 0:
#                             mesh_registered = mesh.transform(self.transformation)
#                             mesh_registered.scale(scale, center=(0, 0, 0))
#                             output_mesh_file = self.file_path.replace('.ply', '_registered_mesh_paraview.ply')
#                             o3d.io.write_triangle_mesh(output_mesh_file, mesh_registered)
#                             saved_files.append("CFD-scaled registered mesh")
#                             summary_message = "Mesh data available for the current .ply file\n"
#                         else:
#                             summary_message = "No mesh data available for the current .ply file\n"
#                     except Exception as e:
#                         print(f"Failed to process mesh data: {str(e)}")
#                         summary_message = "Failed to read mesh data\n"
                
#                 # Use the directory path
#                 directory_path = os.path.dirname(self.file_path)
#                 summary_message += f"Registered data ({', '.join(saved_files)}) saved in the directory: {directory_path}"
#                 self.log_message.emit(summary_message)
#             else:
#                 self.log_message.emit("No registered point cloud available. Perform registration first.")
#         except Exception as e:
#             self.error.emit(str(e))
#         finally:
#             self.finished.emit()




# class MainApp(QMainWindow):
#     def __init__(self):
#         super(MainApp, self).__init__()
#         self.setWindowTitle("FlowVis3D")
#         self.setAcceptDrops(True)
#         self.setupUI()
#         self.registrationThread = None
#         self.registrationWorker = None
#         self.visualizationThread = None
#         self.LoadReferenceWorker = None
#         self.reference_geometry = None
#         self.cached_reference_path = None
#         self.cached_pcd = None  # Cache for the last loaded point cloud
#         self.cached_pcd_path = None  # Path of the last loaded point cloud
#         self.registered_pcd = None
#         self.transformation = None
#         self.workerIsActive = False

#     def setupUI(self):
#         centralWidget = QWidget(self)
#         self.setCentralWidget(centralWidget)
#         layout = QVBoxLayout(centralWidget)

#         # Load and display the logo at the top
#         self.logoLabel = QLabel()
#         self.logoPixmap = QPixmap("./ui/FlowVis3D_logo_v2.jpg").scaled(580, 580, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         # self.logoPixmap = QPixmap("./ui/FlowVis3D_logo_v2.jpg").scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         self.logoLabel.setPixmap(self.logoPixmap)
#         self.logoLabel.setAlignment(Qt.AlignCenter)
#         layout.addWidget(self.logoLabel)

#         # Visualization Section with CollapsibleSection
#         self.visualizationSection = CollapsibleSection("Load", self)
#         layout.addWidget(self.visualizationSection)

#         # File chooser setup for loading point cloud
#         self.loadFileButton = QPushButton("Select Point Cloud")
#         applyButtonStyle(self.loadFileButton)
#         self.visualizationSection.contentLayout().addWidget(self.loadFileButton)
#         # Connect button click to openFileDialog method, passing the line edit as an argument
#         self.loadFileButton.clicked.connect(lambda: self.openFileDialog(self.loadFileLineEdit, "\\\\srvnetapp00\\Technical\\Aerodynamics\\Development\\FlowViz"))

#         # Setup for line edit with rounded edges and placeholder text
#         self.loadFileLineEdit = QLineEdit()
#         self.loadFileLineEdit.setReadOnly(True)
#         self.loadFileLineEdit.setPlaceholderText("Select or drag and drop .ply file")
#         self.loadFileLineEdit.setProperty('role', 'source')
#         self.loadFileLineEdit.setAcceptDrops(True)
#         applyLineEditStyle(self.loadFileLineEdit)
#         self.visualizationSection.contentLayout().addWidget(self.loadFileLineEdit)

#         # Show 3D Point Cloud Button within the visualization section
#         self.visualizeButton = QPushButton("Visualize Point Cloud")
#         applyButtonStyle(self.visualizeButton)
#         self.visualizeButton.clicked.connect(self.startPointCloudVisualization)
#         self.visualizationSection.contentLayout().addWidget(self.visualizeButton)

#          # Spinning wheel setup for visualization
#         self.loadingLabel_visualize, self.loadingMovie_visualize = createLoadingWheel(self.visualizationSection)

#         # Register Section with CollapsibleSection
#         self.registrationSection = CollapsibleSection("Register", self)
#         layout.addWidget(self.registrationSection)

#         # File chooser setup for loading point cloud
#         self.loadRefFileButton = QPushButton("Select Reference")
#         applyButtonStyle(self.loadRefFileButton)
#         self.registrationSection.contentLayout().addWidget(self.loadRefFileButton)
#         # Connect button click to openFileDialog method, passing the line edit as an argument
#         self.loadRefFileButton.clicked.connect(lambda: self.openFileDialog(self.loadRefFileLineEdit, "\\\\srvnetapp00\\Technical\\Aerodynamics\\Development\\FlowViz"))

#         # Allow for drag and drop
#         self.loadRefFileLineEdit = QLineEdit()
#         self.loadRefFileLineEdit.setReadOnly(True)
#         self.loadRefFileLineEdit.setPlaceholderText("Select or drag and drop .ply file")
#         self.loadRefFileLineEdit.setProperty('role', 'reference')
#         self.loadRefFileLineEdit.setAcceptDrops(True)
#         applyLineEditStyle(self.loadRefFileLineEdit)
#         self.registrationSection.contentLayout().addWidget(self.loadRefFileLineEdit)

#         # In the setupUI method of MainApp
#         self.settingsSection = SettingsSection("Settings", self)
#         # Add the SettingsSection to the registration section with left alignment
#         self.registrationSection.contentLayout().addWidget(self.settingsSection)

#         # Execute Registration Button within the registration section
#         self.executeRegistrationButton = QPushButton("Register Point Cloud")
#         applyButtonStyle(self.executeRegistrationButton)
#         self.executeRegistrationButton.clicked.connect(self.executeRegistration)
#         self.registrationSection.contentLayout().addWidget(self.executeRegistrationButton)

#         # Spinning wheel for regkistration operation
#         self.loadingLabel_registration, self.loadingMovie_registration = createLoadingWheel(self.registrationSection)

#        # Registration logs display with combined text area and scrollbar style
#         self.registrationLogLabel = QTextEdit()
#         self.registrationLogLabel.setReadOnly(True)
#         self.registrationLogLabel.setWordWrapMode(QTextOption.WrapAnywhere)
#         self.registrationLogLabel.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show vertical scrollbar only when needed
#         self.registrationLogLabel.setLayoutDirection(Qt.RightToLeft)  # Set the layout direction to RightToLeft
#         self.registrationLogLabel.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show horizontal scrollbar only when needed
#         applyTextAndScrollBarStyle(self.registrationLogLabel)  # Apply combined styles
#         self.registrationLogLabel.setFixedHeight(120)
#         self.registrationSection.contentLayout().addWidget(self.registrationLogLabel)

#         # Button to copy the transformation matrix to the clipboard
#         self.copyMatrixButton = QPushButton("Copy Transformation Matrix")
#         # Style for the copy matrix button
#         applyButtonStyle(self.copyMatrixButton)
#         self.copyMatrixButton.clicked.connect(self.copyTransformationToClipboard)
#         self.registrationSection.contentLayout().addWidget(self.copyMatrixButton)

#         # Visualization Registered Button
#         self.visualizeRegisteredButton = QPushButton("Visualize Registered Point Cloud")
#         applyButtonStyle(self.visualizeRegisteredButton)
#         self.visualizeRegisteredButton.clicked.connect(self.visualizeRegisteredPointCloud)
#         self.registrationSection.contentLayout().addWidget(self.visualizeRegisteredButton)

#         # Save Data
#         self.saveDataButton = QPushButton("Save Data")
#         applyButtonStyle(self.saveDataButton)
#         self.saveDataButton.clicked.connect(self.initiateSaveData)
#         self.registrationSection.contentLayout().addWidget(self.saveDataButton)

#         # Spinning wheel for saving operation
#         self.loadingLabel_savedata, self.loadingMovie_savedata = createLoadingWheel(self.registrationSection)

#         # Registration logs display with combined text area and scrollbar style
#         self.saveLogLabel = QTextEdit()
#         self.saveLogLabel.setReadOnly(True)
#         self.saveLogLabel.setWordWrapMode(QTextOption.WrapAnywhere)
#         self.saveLogLabel.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show vertical scrollbar only when needed
#         self.saveLogLabel.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show horizontal scrollbar only when needed
#         self.saveLogLabel.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show horizontal scrollbar only when needed
#         applyTextAndScrollBarStyle(self.saveLogLabel)  # Apply combined styles
#         self.saveLogLabel.setFixedHeight(75)
#         self.registrationSection.contentLayout().addWidget(self.saveLogLabel)

#         # Preprocessing Section (placeholder)
#         preprocessSection = CollapsibleSection("Pre-process", self)
#         layout.addWidget(preprocessSection)
#         # Populate preprocessSection.contentLayout() as needed in the future

#         # Segmentation Section (placeholder)
#         segmentSection = CollapsibleSection("Segment", self)
#         layout.addWidget(segmentSection)
#         # Populate segmentSection.contentLayout() as needed in the future

#         # Segmentation Section (placeholder)
#         rasterSection = CollapsibleSection("Raster", self)
#         layout.addWidget(rasterSection)
#         # Populate rasterSection.contentLayout() as needed in the future

#         self.show()

# from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QPushButton, QLineEdit, QFileDialog, QLabel, QTextEdit,  QMessageBox, QComboBox, QDialog, QCheckBox, QDialogButtonBox, QHBoxLayout, QToolButton, QRadioButton, QGridLayout, QButtonGroup
# from PySide6.QtGui import QFont, QPixmap, QDragEnterEvent, QDropEvent, QTextOption, QGuiApplication, QIcon
# from PySide6.QtCore import Qt, QThread
# from components.utils import CollapsibleSection, SettingsSection, applyButtonStyle, applyLineEditStyle, applyTextAndScrollBarStyle, applyComboBoxStyle, setupInputField, createLoadingWheel, startLoadingAnimation, stopLoadingAnimation
# from components.workers import LoadPointCloudWorker, LoadMeshWorker, LoadReferenceWorker, RegistrationWorker, SaveWorker, UploadWorker
# import open3d as o3d
# import os

# class MainApp(QMainWindow):
#     """
#     MainApp class that extends QMainWindow, providing the main window for the application.

#     This class handles the initialization and layout of the application's user interface,
#     including setting up the window title, enabling drag-and-drop functionality, and configuring initial state variables and UI components.
#     """

#     def __init__(self):
#         """
#         Constructor for the MainApp class.
#         """
#         super(MainApp, self).__init__()  # Call the constructor of the parent class QMainWindow.
#         self.setWindowTitle("FlowVis3D")  # Set the title of the main window.
#         self.setAcceptDrops(True)  # Enable the main window to accept drag-and-drop events.
#         self.initVariables()  # Initialize application-specific variables and state.
#         self.setupUI()  # Set up the graphical user interface elements of the application.

#     def initVariables(self):
#         """
#         Initialize variables related to application state and workers.
#         """
#         self.registrationThread = None
#         self.registrationWorker = None
#         self.visualizationThread = None
#         self.LoadReferenceWorker = None
#         self.saveThread = None
#         self.saveWorker = None
#         self.uploadThread = None  
#         self.file_path_pcd = None  
#         self.reference_geometry = None
#         self.cached_reference_path = None
#         self.cached_pcd = None
#         self.cached_pcd_path = None
#         self.registered_pcd = None
#         self.registered_mesh = None
#         self.transformation = None
#         self.workerIsActive = False
#         self.case_description = None  
#         self.mesh_upload = None

#     def setupUI(self):
#         """
#         Set up the user interface of the main application window.
#         """
#         centralWidget = QWidget(self)
#         self.setCentralWidget(centralWidget)
#         layout = QVBoxLayout(centralWidget)

#         # Create a layout to hold the reset button and the main content
#         top_layout = QHBoxLayout()
#         layout.addLayout(top_layout)

#         # Add reset button to the top right
#         self.setupResetButton(top_layout)

#         # Add a widget to take the remaining space in the top layout
#         spacer_widget = QWidget()
#         top_layout.addWidget(spacer_widget)
#         top_layout.setStretch(0, 1)  # Make the spacer take up all the extra space

#         self.setupLogo(layout)
#         self.setupSections(layout)
#         self.show()

#     def setupResetButton(self, layout):
#         """
#         Set up the reset button and info icon in the application's user interface.
#         """
#         # Create the reset button
#         self.resetButton = QPushButton()
#         reset_icon = QPixmap("./ui/reset_settings_v2.png")
#         scaled_icon = reset_icon.scaled(25, 25, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         self.resetButton.setIcon(QIcon(scaled_icon))
#         self.resetButton.setIconSize(scaled_icon.rect().size())
#         self.resetButton.setFixedSize(30, 30)
#         self.resetButton.clicked.connect(self.resetSettings)
        
#         # Create the info icon button
#         self.infoButton = QToolButton()
#         self.infoButton.setIcon(QIcon("./ui/info_icon_v2.png"))
#         self.infoButton.setFixedSize(30, 30)
#         self.infoButton.setStyleSheet("QToolButton { border: none; color: white; background-color: black; }")
#         self.infoButton.setToolTip("Clicking the reset button will reset all settings to their default values and clear all logs.")

#         # Add the reset button and info icon to the top-right corner
#         top_right_layout = QHBoxLayout()
#         top_right_layout.addStretch()
#         top_right_layout.addWidget(self.resetButton)
#         top_right_layout.addWidget(self.infoButton)
    
#         # Add the top-right layout to the main layout
#         layout.addLayout(top_right_layout)

#     def resetSettings(self):
#         """
#         Reset all settings and clear logs to reinitialize the application.
#         """
#         # Stop and cleanup any active threads
#         self.cleanupActiveThreads()

#         # Reinitialize variables
#         self.initVariables()
        
#         # Clear all input fields and logs
#         self.loadFileLineEdit.clear()
#         self.loadRefFileLineEdit.clear()
#         self.registrationLogLabel.clear()
#         self.saveLogLabel.clear()
#         self.sandboxLogLabel.clear()
        
#         # Disable buttons that should not be active until new input is provided
#         self.visualizeButton.setDisabled(True)
#         self.visualizeRegisteredButton.setDisabled(True)
        
#         # Optionally, reset the file path and cached data
#         self.file_path_pcd = None
#         self.cached_pcd = None
#         self.cached_pcd_path = None
#         self.reference_geometry = None
#         self.cached_reference_path = None
#         self.registered_pcd = None
#         self.transformation = None
#         self.case_description = None
#         self.mesh_upload = None

#         # # Reset model, WT Run, WT Map, Car Part, and Load Condition fields
#         # self.modelLineEdit.clear()
#         # self.WTRunLineEdit.clear()
#         # self.WTMapLineEdit.clear()
#         # self.carPartComboBox.setCurrentIndex(0)
#         # self.loadConditionComboBox.setCurrentIndex(0)

#         QMessageBox.information(self, "Settings Reset", "All settings have been reset.")

#     def cleanupActiveThreads(self):
#         """
#         Stops and cleans up any active threads and their associated workers.
#         """
#         # Stop and cleanup the registration thread
#         if self.registrationThread and self.registrationThread.isRunning():
#             self.registrationThread.quit()
#             self.registrationThread.wait()
#             self.registrationThread.deleteLater()
#             self.registrationThread = None

#         # Stop and cleanup the visualization thread
#         if self.visualizationThread and self.visualizationThread.isRunning():
#             self.visualizationThread.quit()
#             self.visualizationThread.wait()
#             self.visualizationThread.deleteLater()
#             self.visualizationThread = None

#         # Stop and cleanup the save thread
#         if self.saveThread and self.saveThread.isRunning():
#             self.saveThread.quit()
#             self.saveThread.wait()
#             self.saveThread.deleteLater()
#             self.saveThread = None

#         # Stop and cleanup the upload thread
#         if self.uploadThread and self.uploadThread.isRunning():
#             self.uploadThread.quit()
#             self.uploadThread.wait()
#             self.uploadThread.deleteLater()
#             self.uploadThread = None

#     def setupLogo(self, layout):
#         """
#         Sets up the logo display at the top of the application with dynamic sizing based on the screen resolution.
#         This ensures the logo is appropriately scaled for different display sizes, enhancing visual appearance and usability.
        
#         Args:
#             layout (QLayout): The layout where the logo should be added, typically the main layout of the window.
#         """
#         # Create a QLabel to display the logo.
#         self.logoLabel = QLabel()

#         # Obtain the primary screen's resolution to determine appropriate scaling for the logo.
#         screen = QApplication.primaryScreen()
#         size = screen.size()
#         # Choose the size of the pixmap based on the width of the screen.
#         if size.width() <= 1550:  # Assuming 1550 is typical for smaller, laptop screens
#             pixmap_size = 350
#         else:
#             pixmap_size = 580  # Assuming larger screens have at least this much width

#         # Load the logo image, scale it according to the determined size, and maintain aspect ratio and smooth transformation.
#         logoPixmap = QPixmap("./ui/FlowVis3D_logo_v2.jpg").scaled(pixmap_size, pixmap_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         # Set the scaled pixmap to the label.
#         self.logoLabel.setPixmap(logoPixmap)
#         # Align the logo in the center of the label, which improves layout and appearance.
#         self.logoLabel.setAlignment(Qt.AlignCenter)
#         # Add the logo label to the provided layout.
#         layout.addWidget(self.logoLabel)

#     def setupSections(self, layout):
#         """
#         Setup different collapsible sections of the application.
#         """
#         self.setupVisualizationSection(layout)
#         self.setupRegistrationSection(layout)
#         self.setupSaveSection(layout)
#         self.setupAdditionalRegistrationComponents()
#         self.setupAdditionalSections(layout)

#     def setupVisualizationSection(self, layout):
#         """
#         Sets up the visualization section in the application's user interface.
#         This section is dedicated to loading and viewing point cloud data, specifically in the .ply file format.

#         Args:
#             layout (QVBoxLayout): The layout into which this section is to be integrated.
#         """
#         # Descriptive text that provides information on what the user can do in the "Load" section of the interface.
#         info_text_load = (
#             "Load Section:\n"
#             "\u2022 Select the point cloud (.ply file format).\n"
#             "\u2022 Display the 3D data."
#         )

#         # Create a collapsible section titled "Load" with the descriptive text and add it to the provided layout.
#         self.visualizationSection = CollapsibleSection("Load", self, info_text=info_text_load)
#         layout.addWidget(self.visualizationSection)

#         # Create a button that allows users to select a point cloud file for loading.
#         self.loadFileButton = QPushButton("Select Scanned Point Cloud")
#         applyButtonStyle(self.loadFileButton)  # Apply predefined styling to the button.
#         # Connect the button's click event to open a file dialog, specifying the directory to start in.
#         self.loadFileButton.clicked.connect(lambda: self.openFileDialog(self.loadFileLineEdit, "\\\\srvnetapp00\\Technical\\Aerodynamics\\Development\\FlowViz"))
#         self.visualizationSection.contentLayout().addWidget(self.loadFileButton)  # Add the button to the visualization section's layout.

#         # Create a line edit for displaying the path of the loaded file, which is read-only to prevent manual editing.
#         self.loadFileLineEdit = QLineEdit()
#         self.loadFileLineEdit.setReadOnly(True)
#         self.loadFileLineEdit.setPlaceholderText("Select or drag and drop .ply file")
#         self.loadFileLineEdit.setProperty('role', 'source')  # Custom property to identify the role of the QLineEdit.
#         self.loadFileLineEdit.setAcceptDrops(True)  # Allow drag-and-drop operations directly into the QLineEdit.
#         applyLineEditStyle(self.loadFileLineEdit)  # Apply predefined styling to the line edit.
#         self.visualizationSection.contentLayout().addWidget(self.loadFileLineEdit)  # Add the line edit to the layout.

#         # Create a button to initiate the visualization of the loaded point cloud.
#         self.visualizeButton = QPushButton("Visualize Point Cloud")
#         applyButtonStyle(self.visualizeButton)  # Apply styling.
#         self.visualizeButton.clicked.connect(self.startPointCloudVisualization)  # Connect the click event to the visualization function.
#         self.visualizationSection.contentLayout().addWidget(self.visualizeButton)  # Add the button to the layout.

#         # Create and setup a loading animation indicator, which will be shown while the point cloud is being loaded.
#         self.loadingLabel_visualize, self.loadingMovie_visualize = createLoadingWheel(self.visualizationSection)  # This function returns a label and an animation object.

#     def setupRegistrationSection(self, layout):
#         """
#         Sets up the registration section in the application's user interface.
#         This section is dedicated to managing the registration of point cloud data. It includes functionality for selecting reference files,
#         adjusting registration parameters, and handling registration operations.

#         Args:
#             layout (QVBoxLayout): The layout into which this section is to be integrated.
#         """
#         # Descriptive text that provides information on what the user can do in the "Registration" section of the interface.
#         info_text_registration = (
#             "Registration Section:\n"
#             "\u2022 Select reference point cloud file. Two formats, .ply and .stl, are supported. The .ply format is the standard format. "
#             "If the .stl format is chosen, it will be converted into the .ply format for the registration procedure, resulting in additional computational time.\n"
#             "\u2022 Align point clouds using the registration algorithm (RANSAC Global Registration and ICP). Due to the randomic nature of the algorithm, it may be necessary to "
#             "repeat the registration procedure a second time to obtain satisfying results.\n"
#             "\u2022 Adjust registration settings such as voxel size and desired accuracy (optional).\n"
#             "\u2022 Copy transformation matrix.\n"
#             "\u2022 Display registered point cloud for inspection.\n"
#         )

#         # Create a collapsible section titled "Register" with the descriptive text and add it to the provided layout.
#         self.registrationSection = CollapsibleSection("Register", self, info_text=info_text_registration)
#         layout.addWidget(self.registrationSection)

#         # Create a button that allows users to select a reference point cloud file for registration.
#         self.loadRefFileButton = QPushButton("Select Reference")
#         applyButtonStyle(self.loadRefFileButton)  # Apply predefined styling to the button.
#         # Connect the button's click event to open a file dialog, specifying the directory to start in.
#         self.loadRefFileButton.clicked.connect(lambda: self.openFileDialog(self.loadRefFileLineEdit, "\\\\srvnetapp00\\Technical\\Aerodynamics\\Development\\FlowViz"))
#         self.registrationSection.contentLayout().addWidget(self.loadRefFileButton)  # Add the button to the registration section's layout.

#         # Create a line edit for displaying the path of the loaded reference file, which is read-only to prevent manual editing.
#         self.loadRefFileLineEdit = QLineEdit()
#         self.loadRefFileLineEdit.setReadOnly(True)
#         self.loadRefFileLineEdit.setPlaceholderText("Select or drag and drop .ply file")
#         self.loadRefFileLineEdit.setProperty('role', 'reference')  # Custom property to identify the role of the QLineEdit.
#         self.loadRefFileLineEdit.setAcceptDrops(True)  # Allow drag-and-drop operations directly into the QLineEdit.
#         applyLineEditStyle(self.loadRefFileLineEdit)  # Apply predefined styling to the line edit.
#         self.registrationSection.contentLayout().addWidget(self.loadRefFileLineEdit)  # Add the line edit to the layout.

#         # Add a settings section for adjusting registration parameters like voxel size and accuracy.
#         self.settingsSection = SettingsSection("Settings", self)
#         self.registrationSection.contentLayout().addWidget(self.settingsSection)

#         # Create a button to initiate the registration of the point cloud.
#         self.executeRegistrationButton = QPushButton("Register Point Cloud")
#         applyButtonStyle(self.executeRegistrationButton)  # Apply styling.
#         self.executeRegistrationButton.clicked.connect(self.executeRegistration)  # Connect the click event to the registration function.
#         self.registrationSection.contentLayout().addWidget(self.executeRegistrationButton)  # Add the button to the layout.

#         # Create and setup a loading animation indicator, which will be shown while the registration process is running.
#         self.loadingLabel_registration, self.loadingMovie_registration = createLoadingWheel(self.registrationSection)  # This function returns a label and an animation object.

#         # Setup a logging area for registration operations.
#         self.setupRegistrationLog()  # This method configures a log area to display registration process messages.

#     def setupRegistrationLog(self):
#         """
#         Setup the log area for registration operations.
#         """
#         self.registrationLogLabel = QTextEdit()
#         self.registrationLogLabel.setReadOnly(True)
#         self.registrationLogLabel.setWordWrapMode(QTextOption.WrapAnywhere)
#         self.registrationLogLabel.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
#         self.registrationLogLabel.setLayoutDirection(Qt.RightToLeft)
#         applyTextAndScrollBarStyle(self.registrationLogLabel)
#         self.registrationLogLabel.setFixedHeight(120)
#         self.registrationSection.contentLayout().addWidget(self.registrationLogLabel)

#     def setupAdditionalRegistrationComponents(self):
#         """
#         Configures additional interactive components within the registration section of the user interface.
#         This method adds functionality to copy transformation matrices, visualize and save registered point clouds,
#         and manage data upload operations with visual feedback for each process.
#         """

#         # Create a button that allows users to copy the transformation matrix to the clipboard.
#         self.copyMatrixButton = QPushButton("Copy Transformation Matrix")
#         applyButtonStyle(self.copyMatrixButton)  # Apply predefined styling to the button.
#         self.copyMatrixButton.clicked.connect(self.copyTransformationToClipboard)  # Connect the button's click event to the copy functionality.
#         self.registrationSection.contentLayout().addWidget(self.copyMatrixButton)  # Add the button to the registration section's layout.

#         # Create a button to initiate the visualization of the registered point cloud.
#         self.visualizeRegisteredButton = QPushButton("Visualize Registered Point Cloud")
#         applyButtonStyle(self.visualizeRegisteredButton)  # Apply styling.
#         self.visualizeRegisteredButton.clicked.connect(self.visualizeRegisteredPointCloud)  # Connect the click event to the visualization function.
#         self.registrationSection.contentLayout().addWidget(self.visualizeRegisteredButton)  # Add the button to the layout.

#     def setupSaveSection(self, layout):
#         """
#         Sets up the save section in the application's user interface.
#         This section is intended to store registered point cloud data and upload it to sandbox. It includes functionality for setting the registered file name correctly,
#         according to the current data nomenclature, saving it in the network drive and handling automatically the upload to sandbox.

#         Args:
#             layout (QVBoxLayout): The layout into which this section is to be integrated.
#         """
#         # Descriptive text that provides information on what the user can do in the "Registration" section of the interface.
#         info_text_save = (
#             "Save Section:\n"
#             "\u2022 Provide input parameters for correct file naming, according to the current nomenclature.\n"
#             "\u2022 Save the registered data: save the registered wind tunnel model 60\u0025 scale point cloud ('_registered.ply'), the up-scaled registered point cloud ('_registered_paraview.ply') "
#             "and, if available, wind tunnel model 60\u0025 scale mesh ('_registered_mesh.ply') and the up-scaled registered mesh ('_registered_mesh_paraview.ply') in the input point cloud original folder. The files are named according to the sandbox nomenclature"
#             "The up-scaled data is intended for paraview/sandbox, where one can compare FlowVis with CFD data (100\u0025 scale).\n"
#             "\u2022 Upload to Sandbox: according to the provided session parameters, the user can upload automatically the scanned mesh to Sandbox. If a registered version of the input mesh is available, this is will be the item uploaded to Sandbox." # ADD BUTTON TO ASK USER WHAT TO UPLOAD
#         )

#         # Create a collapsible section titled "Save" with the descriptive text and add it to the provided layout.
#         self.saveSection = CollapsibleSection("Save", self, info_text=info_text_save)
#         layout.addWidget(self.saveSection)

#         # Create a label with the text "Enter WT Run Data" and add it to the provided layout.
#         headerLabel = QLabel("Enter WT Run Data")
#         headerLabel.setAlignment(Qt.AlignLeft)  # Center-align the text
#         font = QFont()
#         font.setBold(True)
#         headerLabel.setFont(font)
#         self.saveSection.contentLayout().addWidget(headerLabel)  # Add the button to the layout.

#         # Fetch WT Run data
#         self.modelLineEdit = setupInputField(self.saveSection.contentLayout(), "Model", "M44")
#         self.WTRunLineEdit = setupInputField(self.saveSection.contentLayout(), "WT Run", "4748")
#         self.WTMapLineEdit = setupInputField(self.saveSection.contentLayout(), "WT Map", "STD_MAP_056d_v258_LS_03") 

#         # Add "Car Part" label
#         carPartLabel = QLabel("Car Part")
#         carPartLabel.setAlignment(Qt.AlignLeft) 
#         self.saveSection.contentLayout().addWidget(carPartLabel)

#         # Add dropdown menu for car parts
#         self.carPartComboBox = QComboBox()
#         self.carPartComboBox.addItems(["body", "front brake", "rear brake", "floor", "front wheel", "rear wheel", "front wing", "rear wing", "internal", "front suspension", "rear suspension"])
#         applyComboBoxStyle(self.carPartComboBox)
#         self.saveSection.contentLayout().addWidget(self.carPartComboBox)

#         # Dictionary for correspondences
#         self.car_part_correspondences = {
#             "body": "bdy",
#             "front brake": "brk-frt",
#             "rear brake": "brk-rr",
#             "floor": "floor",
#             "front wheel": "frt-whl",
#             "rear wheel": "rr-whl",
#             "front wing": "fw",
#             "rear wing": "rw",
#             "internal": "intnl",
#             "front suspension": "susp-frt",
#             "rear suspension": "susp-rr"
#         }

#         # Add "Load Condition" label
#         loadConditionLabel = QLabel("Load Condition")
#         loadConditionLabel.setAlignment(Qt.AlignLeft) 
#         self.saveSection.contentLayout().addWidget(loadConditionLabel)

#         # Add dropdown menu for load conditions
#         self.loadConditionComboBox = QComboBox()
#         self.loadConditionComboBox.addItems(["loaded", "unloaded", "straightline"])
#         applyComboBoxStyle(self.loadConditionComboBox)
#         self.saveSection.contentLayout().addWidget(self.loadConditionComboBox)

#         # Add "Notes" label
#         notesLabel = QLabel("Notes (Optional)")
#         notesLabel.setAlignment(Qt.AlignLeft)
#         self.saveSection.contentLayout().addWidget(notesLabel)

#         # Add QLineEdit for notes
#         self.notesLineEdit = QLineEdit()
#         applyLineEditStyle(self.notesLineEdit)
#         self.saveSection.contentLayout().addWidget(self.notesLineEdit)

#         # Add "File Name" label
#         filenameLabel = QLabel("File Name")
#         filenameLabel.setAlignment(Qt.AlignLeft) 
#         self.saveSection.contentLayout().addWidget(filenameLabel)
#         # Add QLineEdit for displaying the file name
#         self.fileNameLineEdit = QLineEdit()
#         self.fileNameLineEdit.setReadOnly(True)
#         applyLineEditStyle(self.fileNameLineEdit)
#         self.saveSection.contentLayout().addWidget(self.fileNameLineEdit)

#         # Connect signals to update the file name when any input changes
#         self.modelLineEdit.textChanged.connect(self.updateFileName)
#         self.WTRunLineEdit.textChanged.connect(self.updateFileName)
#         self.WTMapLineEdit.textChanged.connect(self.updateFileName)
#         self.carPartComboBox.currentIndexChanged.connect(self.updateFileName)
#         self.loadConditionComboBox.currentIndexChanged.connect(self.updateFileName)
#         self.notesLineEdit.textChanged.connect(self.updateFileName)

#         # Create a button to initiate the saving of the registered point cloud data.
#         self.saveDataButton = QPushButton("Save Data")
#         applyButtonStyle(self.saveDataButton)  # Apply styling.
#         self.saveDataButton.clicked.connect(self.initiateSaveData)  # Connect the click event to the save data function.
#         self.saveSection.contentLayout().addWidget(self.saveDataButton)  # Add the button to the layout.

#         # Create and setup a loading animation indicator for the saving operation, which will be shown while data is being saved.
#         self.loadingLabel_savedata, self.loadingMovie_savedata = createLoadingWheel(self.saveSection)  # This function returns a label and an animation object.

#         # Setup a text edit widget for logging save operation messages.
#         self.saveLogLabel = QTextEdit()
#         self.saveLogLabel.setReadOnly(True)  # Make the log read-only.
#         self.saveLogLabel.setWordWrapMode(QTextOption.WrapAnywhere)  # Enable word wrapping.
#         self.saveLogLabel.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Add vertical scroll bar if needed.
#         self.saveLogLabel.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Add horizontal scroll bar if needed.
#         applyTextAndScrollBarStyle(self.saveLogLabel)  # Apply predefined styling.
#         self.saveLogLabel.setFixedHeight(75)  # Set a fixed height for the log display.
#         self.saveSection.contentLayout().addWidget(self.saveLogLabel)  # Add the log display to the layout.

#         # Create a button for uploading registered data to a sandbox environment.
#         self.uploadSandboxButton = QPushButton("Upload to Sandbox")
#         applyButtonStyle(self.uploadSandboxButton)  # Apply styling.
#         self.uploadSandboxButton.clicked.connect(self.uploadtoSandbox)  # Connect the click event to the upload function.
#         self.saveSection.contentLayout().addWidget(self.uploadSandboxButton)  # Add the button to the layout.

#         # Create and setup a loading animation indicator for the upload operation, which will be shown while data is being uploaded.
#         self.loadingLabel_upload, self.loadingMovie_upload = createLoadingWheel(self.saveSection)  

#         # Setup a text edit widget for logging upload operation messages.
#         self.sandboxLogLabel = QTextEdit()
#         self.sandboxLogLabel.setReadOnly(True)  # Make the log read-only.
#         self.sandboxLogLabel.setWordWrapMode(QTextOption.WrapAnywhere)  # Enable word wrapping.
#         self.sandboxLogLabel.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Add vertical scroll bar if needed.
#         self.sandboxLogLabel.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Add horizontal scroll bar if needed.
#         applyTextAndScrollBarStyle(self.sandboxLogLabel)  # Apply predefined styling.
#         self.sandboxLogLabel.setFixedHeight(75)  # Set a fixed height for the log display.
#         self.saveSection.contentLayout().addWidget(self.sandboxLogLabel)  # Add the log display to the layout.


#     def updateFileName(self):
#         """
#         Updates the file name based on user inputs and displays it in the QLineEdit.
#         """
#         model = self.modelLineEdit.text()
#         wt_run = self.WTRunLineEdit.text()
#         wt_map = self.WTMapLineEdit.text()
#         car_part = self.car_part_correspondences[self.carPartComboBox.currentText()]
#         load_condition = self.loadConditionComboBox.currentText()
#         notes = self.notesLineEdit.text().strip()

#         # Extract case description from wt_map
#         self.case_description = None
#         case_descriptions = ["SL_01", "LS_02", "LS_03", "VHS_04", "LS_05", "MS_06", "MS_07", "MS_09", "LS_09", "LS_10"]
#         for case in case_descriptions:
#             if case in wt_map:
#                 self.case_description = case
#                 break

#         # if self.case_description:
#         #     print(f"Case Description: {self.case_description}")

#         if notes:
#             file_name = f"{model}_{wt_run}_FV_Sauber_{wt_map}_{car_part}_{load_condition}_{notes}"
#         else:
#             file_name = f"{model}_{wt_run}_FV_Sauber_{wt_map}_{car_part}_{load_condition}"

#         self.fileNameLineEdit.setText(file_name)

#     def setupAdditionalSections(self, layout):
#         """
#         Setup additional sections such as Pre-process, Segment, and Raster (currently place-holders).
#         """
#         info_text_preprocessing = (
#             "Preprocessing Section (work in progress):\n"
#             "\u2022 Downsample the input registered point cloud.\n"
#             "\u2022 Estimate the point cloud normals."
#         )
#         info_text_segmentation = (
#             "Segmentation Section (work in progress):\n"
#             "\u2022 Divide the point cloud into coarse surfaces using K-means.\n"
#             "\u2022 Segment the point cloud into coherent surfaces using DBSCAN."
#         )
#         info_text_raster = (
#             "Raster Section (work in progress):\n"
#             "\u2022 Orient each point cloud segment using PCA.\n"
#             "\u2022 Project each PCA-rotated segmented onto an optimal 2D plane and obtain a 2D representation of the segment."
#         )
#         preprocessSection = CollapsibleSection("Pre-process", self, info_text=info_text_preprocessing)
#         segmentSection = CollapsibleSection("Segment", self, info_text=info_text_segmentation)
#         rasterSection = CollapsibleSection("Raster", self, info_text=info_text_raster)
#         layout.addWidget(preprocessSection)
#         layout.addWidget(segmentSection)
#         layout.addWidget(rasterSection)

#     def openFileDialog(self, lineEdit, initial_dir):
#         """
#         Opens a file dialog allowing the user to select a point cloud file from the filesystem.
#         This method updates the provided QLineEdit with the selected file path and adjusts the application state based on the selection.

#         Args:
#             lineEdit (QLineEdit): The QLineEdit widget to update with the selected file path.
#             initial_dir (str): The initial directory to open in the file dialog.
#         """
#         # Create a file dialog object with a specific title and parent (self refers to the MainApp class).
#         dialog = QFileDialog(self, "Select Point Cloud File")
#         dialog.setDirectory(initial_dir)  # Set the initial directory of the file dialog.
#         dialog.setNameFilter("Point Cloud Files (*.ply *.stl);;All Files (*)")  # Set the file type filters.
#         dialog.setFileMode(QFileDialog.ExistingFile)  # Allow selection of only existing files.
#         dialog.setViewMode(QFileDialog.Detail)  # Set the view mode to show detailed information about files.
#         dialog.setModal(True)  # Make the dialog modal, blocking interactions with other windows until it is closed.
#         dialog.resize(800, 600)  # Set the initial size of the dialog.
#         dialog.setMinimumSize(800, 600)  # Set the minimum size of the dialog to prevent resizing to a smaller window.
#         dialog.setWindowState(Qt.WindowNoState)  # Ensure the window state is normal (not minimized, maximized, or fullscreen).

#         # Execute the dialog and check if the user accepted (clicked OK).
#         if dialog.exec() == QFileDialog.Accepted:
#             selectedFiles = dialog.selectedFiles()  # Retrieve the list of selected files.
#             if selectedFiles:  # Check if there is at least one selected file.
#                 file_name = selectedFiles[0]  # Get the first file from the list.
#                 lineEdit.setText(file_name)  # Update the QLineEdit widget with the selected file path.

#                 # Specific behavior if the line edit is for loading point clouds, enables visualization button.
#                 if lineEdit is self.loadFileLineEdit:
#                     self.file_path_pcd = file_name  # Update the stored file path for the point cloud.
#                     self.visualizeButton.setDisabled(False)  # Enable the visualization button when a file is selected.

#                 # Specific behavior if the line edit is for loading reference files in registration, enables registration button.
#                 if lineEdit is self.loadRefFileLineEdit:
#                     self.visualizeRegisteredButton.setDisabled(False)  # Enable the registration button when a reference is selected.

#     def dragEnterEvent(self, event: QDragEnterEvent):
#         """
#         Handles the event when a drag operation enters the application window. This method is triggered
#         whenever a drag operation involves the application's GUI.

#         Args:
#             event (QDragEnterEvent): Contains information about the drag event, including the data being dragged.
#         """
#         # Check if the drag data includes URLs, which typically represent file paths when dragging files from the file explorer.
#         if event.mimeData().hasUrls():
#             # If there are URLs, accept the drag action. This allows the dropEvent method to be called if the user drops the files.
#             event.acceptProposedAction()

#     def dropEvent(self, event: QDropEvent):
#         """
#         Handles the drop event when a user releases a dragged item (e.g., file) over the application window.
#         This function specifically updates text fields based on the dropped file's path, depending on where it was dropped.

#         Args:
#             event (QDropEvent): Contains information about the drop action, including the data and the position of the drop.
#         """
#         # Extract the URLs from the drop event's MIME data. URLs represent the paths of the files being dropped.
#         urls = event.mimeData().urls()
#         if urls:  # Check if there are any URLs
#             path = urls[0].toLocalFile()  # Convert the first URL to a local file path.
#             cursor_position = event.position()  # Get the position of the cursor at the time of the drop.

#             # Determine which widget is directly beneath the cursor at the drop position.
#             child = self.childAt(cursor_position.toPoint())
#             if child:
#                 # If the widget directly under the cursor isn't a QLineEdit, traverse up the widget hierarchy to find the nearest parent QLineEdit.
#                 while not isinstance(child, QLineEdit) and child is not None:
#                     child = child.parent()
#                 if child and isinstance(child, QLineEdit):  # Check if a QLineEdit was found.
#                     # Retrieve a custom property ('role') from the QLineEdit to determine its intended role (source or reference).
#                     role = child.property('role')
#                     if role == 'source':
#                         self.loadFileLineEdit.setText(path)  # Update the 'source' line edit with the path of the dropped file.
#                         self.file_path_pcd = path  # Update the file_path_pcd for saving later.
#                         self.visualizeButton.setDisabled(False)  # Enable the visualize button
#                     elif role == 'reference':
#                         self.loadRefFileLineEdit.setText(path)  # Update the 'reference' line edit with the path.

#     def getFilePath(self):
#         """
#         Retrieves the current text from the fileLineEdit widget, which typically holds the path to a selected file.

#         Returns:
#             str: The text content of the fileLineEdit, representing the file path.
#         """
#         # Return the text content of the fileLineEdit, which is expected to be a file path.
#         return self.fileLineEdit.text()

#     def setLoadFilePath(self, file_path):
#         """
#         Sets the text of the filePathLineEdit widget to the specified file path. This is typically used
#         to update the UI with a new path after a file has been selected or changed.

#         Args:
#             file_path (str): The file path to set in the filePathLineEdit widget.
#         """
#         # Update the filePathLineEdit with the new file path, reflecting changes or new selections in the UI.
#         self.filePathLineEdit.setText(file_path)

#     def setReferenceFilePath(self, file_path):
#         """
#         Sets the text of the referenceFilePathLineEdit widget to the specified reference file path. This method is used
#         to update the path displayed in the referenceFilePathLineEdit, typically after selecting a new reference file.

#         Args:
#             file_path (str): The reference file path to set in the referenceFilePathLineEdit widget.
#         """
#         # Update the referenceFilePathLineEdit with the new reference file path, reflecting the user's selection or changes.
#         self.referenceFilePathLineEdit.setText(file_path)


#     def startPointCloudVisualization(self):
#         """
#         Initiates the process of loading and visualizing a point cloud based on the file path provided in the loadFileLineEdit.
#         This method checks if the file path has changed or if the point cloud is not already loaded before starting the loading process.
#         """
#         # Retrieve the file path from the loadFileLineEdit widget.
#         file_path = self.loadFileLineEdit.text()
        
#         # Check if a file path is not specified.
#         if not file_path:
#             QMessageBox.warning(self, "Warning", "No file selected. Please load a point cloud first.")
#             return

#         # Check if the point cloud needs to be reloaded (path change or not previously loaded).
#         if self.cached_pcd_path != file_path or self.cached_pcd is None:
#             # Start the loading animation to indicate to the user that processing is underway.
#             startLoadingAnimation(self.loadingLabel_visualize, self.loadingMovie_visualize, self.visualizationSection.scrollArea)

#             # Initialize a worker and thread for loading the point cloud from the specified file path.
#             self.loadPointCloudWorker = LoadPointCloudWorker(file_path=file_path)
#             self.loadPointCloudThread = QThread()
#             self.loadPointCloudWorker.moveToThread(self.loadPointCloudThread)

#             # Connect the finished signals of the worker to the appropriate slots.
#             self.loadPointCloudWorker.finished.connect(self.handleLoadPointCloudComplete)
#             self.loadPointCloudWorker.finished.connect(self.loadPointCloudThread.quit)
#             self.loadPointCloudWorker.finished.connect(self.loadPointCloudWorker.deleteLater)
#             self.loadPointCloudThread.finished.connect(self.loadPointCloudThread.deleteLater)

#             # Start the thread, which triggers the worker's run method.
#             self.loadPointCloudThread.started.connect(self.loadPointCloudWorker.run)
#             self.loadPointCloudThread.start()
#         else:
#             # If the point cloud is already loaded and the path hasn't changed, directly visualize the cached point cloud.
#             self.visualizePointCloud(self.cached_pcd, "")

#     def handleLoadPointCloudComplete(self, pcd, error_message):
#         """
#         Handles the completion of the point cloud loading process. This method is called when the LoadPointCloudWorker
#         has finished processing, and it manages both successful and failed loads.

#         Args:
#             pcd: The loaded point cloud object or None if loading failed.
#             error_message (str): A message describing the error if one occurred during loading.
#         """
#         # Stop the loading animation regardless of success or error.
#         stopLoadingAnimation(self.loadingLabel_visualize, self.loadingMovie_visualize, self.visualizationSection.scrollArea)

#         # Handle errors during point cloud loading.
#         if error_message:
#             QMessageBox.critical(self, "Visualization Error", error_message)
#             return

#         # If a point cloud was successfully loaded, proceed with visualization.
#         if pcd:
#             # Cache the successfully loaded point cloud and its path.
#             self.cached_pcd = pcd
#             self.cached_pcd_path = self.loadFileLineEdit.text()

#             # Attempt to visualize the loaded point cloud using Open3D.
#             try:
#                 o3d.visualization.draw_geometries([pcd], window_name='FlowVis3D')
#             except Exception as e:
#                 QMessageBox.critical(self, "Visualization Error", f"Failed to visualize the point cloud: {str(e)}")
#         else:
#             # Alert the user if no point cloud was loaded.
#             QMessageBox.warning(self, "Load Warning", "No point cloud was loaded.")

#     def visualizePointCloud(self, pcd, error_message):
#         """
#         Visualizes a given point cloud if no error occurred during the loading or processing stages.
#         This method attempts to render the point cloud using Open3D visualization tools.

#         Args:
#             pcd: The point cloud data to visualize. This is typically an object compatible with Open3D visualization.
#             error_message (str): An error message indicating if something went wrong before attempting visualization.
#         """
#         # Check if there is an error message and display it if present.
#         if error_message:
#             QMessageBox.critical(self, "Visualization Error", error_message)
#             # Uncomment below to stop loading animation if used elsewhere in the application context.
#             # stopLoadingAnimation(self.loadingLabel_visualize, self.loadingMovie_visualize, self.visualizationSection.scrollArea)
#             return

#         # If the point cloud data is not None, attempt to visualize it.
#         if pcd is not None:
#             try:
#                 # Use Open3D to draw geometries; this function will open a visualization window.
#                 o3d.visualization.draw_geometries([pcd], window_name='FlowVis3D')
#             except Exception as e:
#                 # If visualization fails, catch the exception and show an error message.
#                 QMessageBox.critical(self, "Visualization Error", f"Failed to visualize the point cloud: {str(e)}")

#         # If applicable, stop any ongoing animations that indicate loading or processing.
#         # stopLoadingAnimation(self.loadingLabel_visualize, self.loadingMovie_visualize, self.visualizationSection.scrollArea)

#     def handleLoadError(self, error_message):
#         """
#         Handles errors that may occur during the loading of point cloud data.
#         This method displays an error message and stops any visual indications of ongoing processes, such as animations.

#         Args:
#             error_message (str): A message describing the error that occurred during the loading process.
#         """
#         # Display a critical error message using a message box.
#         QMessageBox.critical(self, "Load Error", f"Failed to load the point cloud: {error_message}")
#         # Stop any loading animations to indicate that the process has ended, possibly due to an error.
#         stopLoadingAnimation(self.loadingLabel_visualize, self.loadingMovie_visualize)

#     def executeRegistration(self):
#         """
#         Initiates the point cloud registration process by setting up and starting a new worker thread.
#         This method also handles user input validation and manages UI updates related to the registration process.
#         """
#         # Retrieve the registration settings from the settings section.
#         voxel_sizes, desired_fitness_ransac, desired_fitness_icp = self.settingsSection.getSettings() 

#         # Clear any previous messages in the registration log to prepare for new output.
#         self.registrationLogLabel.clear()

#         # Ensure that any previously running registration threads are properly terminated before starting a new one.
#         if self.registrationThread is not None:
#             if self.registrationThread.isRunning():
#                 self.registrationThread.quit()
#                 self.registrationThread.wait()

#         # Start the loading animation in the UI to indicate that registration is processing.
#         startLoadingAnimation(self.loadingLabel_registration, self.loadingMovie_registration, self.registrationSection.scrollArea)

#         # Fetch the file paths for the source and reference point clouds from the respective QLineEdit widgets.
#         sourceFilePath = self.loadFileLineEdit.text()
#         referenceFilePath = self.loadRefFileLineEdit.text()

#         # Check if both necessary files are selected, and show a warning if either is missing.
#         if not sourceFilePath or not referenceFilePath:
#             missing_files = []
#             if not sourceFilePath:
#                 missing_files.append("point cloud")
#             if not referenceFilePath:
#                 missing_files.append("reference")
#             missing = " and ".join(missing_files)
#             QMessageBox.warning(self, "Warning", f"Please load {missing}.")
#             stopLoadingAnimation(self.loadingLabel_registration, self.loadingMovie_registration, self.registrationSection.scrollArea)
#             return

#         # Initialize the registration worker with the file paths and registration settings.
#         self.registrationWorker = RegistrationWorker(sourceFilePath, referenceFilePath, self, voxel_sizes, desired_fitness_ransac, desired_fitness_icp=desired_fitness_icp)
#         self.registrationThread = QThread()  # Create a new thread for registration operations.
#         self.registrationWorker.moveToThread(self.registrationThread)  # Move the worker to the new thread.

#         # Connect the signals from the worker to appropriate slots to handle completion and errors.
#         self.registrationWorker.finished.connect(self.handleRegistrationComplete)
#         self.registrationWorker.error.connect(self.handleRegistrationError)

#         # Print to console for debugging purposes indicating that signal connections are established.
#         # print("Connecting signals")

#         # Start the worker thread and handle its lifecycle.
#         self.registrationThread.started.connect(self.registrationWorker.run)
#         self.registrationThread.finished.connect(self.registrationThread.deleteLater)
#         self.registrationThread.start()  # Begin the registration process.

#     def handleRegistrationComplete(self, pcd_registered, transformation, log_text, pcd_registered_scaled, registered_mesh, registered_mesh_scaled):
#         """
#         Handles the successful completion of the registration process. Updates the UI and caches the results.

#         Args:
#             pcd_registered: The registered point cloud returned by the registration worker.
#             transformation: The transformation matrix resulting from the registration.
#             log_text (str): Text to log in the UI, typically containing details about the registration process.
#             pcd_registered_scaled: The scaled registered point cloud.
#             registered_mesh: The registered mesh (if available).
#             registered_mesh_scaled: The scaled registered mesh (if available).
#         """
#         stopLoadingAnimation(self.loadingLabel_registration, self.loadingMovie_registration, self.registrationSection.scrollArea)
#         self.registrationLogLabel.setText(log_text)

#         self.registered_pcd = pcd_registered
#         self.transformation = transformation
#         self.registered_pcd_scaled = pcd_registered_scaled
#         self.registered_mesh = registered_mesh
#         self.registered_mesh_scaled = registered_mesh_scaled

#         if registered_mesh is None:
#             QMessageBox.warning(self, "Registration Complete", "No mesh data available. Upload to Sandbox is not possible.")

#     def handleRegistrationError(self, error_message):
#         """
#         Handles errors that occur during the registration process. This method updates the UI to reflect the error and stops any loading animations.

#         Args:
#             error_message (str): A message describing what went wrong during registration.
#         """
#         # Stop any ongoing loading animations to indicate that the registration process has halted due to an error.
#         stopLoadingAnimation(self.loadingLabel_registration, self.loadingMovie_registration, self.registrationSection.scrollArea)

#         # Update the registration log in the UI with the error message to inform the user what went wrong.
#         self.registrationLogLabel.setText(f"Registration failed: {error_message}")

#     def copyTransformationToClipboard(self):
#         """
#         Copies the transformation matrix, if available, to the system clipboard. This allows the user to paste it elsewhere.

#         """
#         # Access the system clipboard through the QGuiApplication.
#         clipboard = QGuiApplication.clipboard()
#         # Convert the transformation matrix to string format if it exists and copy it to the clipboard.
#         transformation_text = str(self.transformation)  # Ensure this contains the transformation matrix text.
#         clipboard.setText(transformation_text)
#         # Print to console for debugging purposes to confirm that the matrix has been copied.
#         # print("Transformation matrix copied to clipboard.")

#     def visualizeRegisteredPointCloud(self):
#         """
#         Visualizes the registered point cloud if it's available and if the reference file necessary for comparison is loaded.
#         """
#         # Print to console for debugging purposes about the visualization attempt.
#         # print("Visualizing registered point cloud. Current data:", self.registered_pcd)

#         # Check if there's a registered point cloud available to visualize.
#         if not self.registered_pcd:
#             QMessageBox.warning(self, "Visualization Error", "No registered point cloud available. Please perform registration first.")
#             return

#         # Check if the reference file path is loaded.
#         current_path = self.loadRefFileLineEdit.text().strip()
#         if not current_path:
#             QMessageBox.warning(self, "Visualization Error", "No reference file loaded. Please load a reference file.")
#             return

#         # Determine if the geometry needs to be reloaded due to a change in the reference path or if it's not loaded.
#         if current_path != self.cached_reference_path or self.reference_geometry is None:
#             # Reset the reference geometry and update the cached path.
#             self.reference_geometry = None
#             self.cached_reference_path = current_path

#             # Clean up any existing visualization infrastructure before setting up a new load operation.
#             self.tearDownVisualizationInfrastructure()

#             # Set up a new worker to load the reference geometry based on the current path.
#             self.setupLoadReferenceWorker(current_path)
#         else:
#             # If the geometry is already loaded and the path hasn't changed, proceed to visualize it directly.
#             self.visualizeData(self.reference_geometry)

#     def setupLoadReferenceWorker(self, path):
#         """
#         Sets up a worker and a thread to load reference geometry from the specified path. This method ensures
#         that any previous workers and threads are properly cleaned up before initiating a new load operation.

#         Args:
#             path (str): The file path from which the reference geometry is to be loaded.
#         """
#         # First, clean up any existing visualization infrastructure to avoid conflicts.
#         self.tearDownVisualizationInfrastructure()

#         # Create a new worker for loading the reference geometry.
#         self.LoadReferenceWorker = LoadReferenceWorker(path)
#         self.visualizationThread = QThread()  # Create a new thread.

#         # Move the newly created worker to the new thread.
#         self.LoadReferenceWorker.moveToThread(self.visualizationThread)

#         # Connect various signals of the worker to appropriate slots.
#         # These include data loaded successfully, errors, and cleanup signals.
#         self.LoadReferenceWorker.dataLoaded.connect(self.cacheAndVisualizeData)
#         self.LoadReferenceWorker.error.connect(self.handleVisualizationError)
#         self.LoadReferenceWorker.finished.connect(self.LoadReferenceWorker.deleteLater)
#         self.LoadReferenceWorker.finished.connect(self.visualizationThread.quit)

#         # Start the thread which in turn starts the worker's run method.
#         self.visualizationThread.started.connect(self.LoadReferenceWorker.run)
#         self.visualizationThread.finished.connect(self.visualizationThread.deleteLater)
#         self.visualizationThread.start()

#     def tearDownVisualizationInfrastructure(self):
#         """
#         Cleans up any existing visualization workers and threads to ensure a clean state before starting new operations.
#         """
#         # Check if there is an existing LoadReferenceWorker and if it is active.
#         if self.LoadReferenceWorker:
#             if self.LoadReferenceWorker.active:
#                 # Attempt to disconnect any connected signals to prevent memory leaks and ensure proper cleanup.
#                 try:
#                     self.LoadReferenceWorker.dataLoaded.disconnect()
#                     self.LoadReferenceWorker.error.disconnect()
#                     self.LoadReferenceWorker.finished.disconnect()
#                 except RuntimeError as e:
#                     print(f"Error while disconnecting signals: {e}")
#             # Safely delete the worker.
#             self.LoadReferenceWorker.deleteLater()
#             self.LoadReferenceWorker = None

#         # Check if there is an existing thread for visualization and ensure it is properly terminated.
#         if self.visualizationThread:
#             if self.visualizationThread.isRunning():
#                 self.visualizationThread.quit()
#                 self.visualizationThread.wait()
#             self.visualizationThread.deleteLater()
#             self.visualizationThread = None

#     def cacheAndVisualizeData(self, reference_geom):
#         """
#         Caches the loaded reference geometry and triggers its visualization. This method is typically called
#         when the reference geometry has been successfully loaded by a worker.

#         Args:
#             reference_geom: The reference geometry that has been loaded and is now ready for visualization.
#         """
#         # Cache the loaded reference geometry for future use or re-use without reloading.
#         self.reference_geometry = reference_geom
#         # Call the method to visualize the reference geometry now stored in cache.
#         self.visualizeData(reference_geom)

#     def visualizeData(self, reference_geom):
#         """
#         Visualizes the loaded reference geometry alongside the registered point cloud using Open3D visualization tools.
#         This method attempts to set up a visualization environment, add the geometries to it, and display them.

#         Args:
#             reference_geom: The geometry to be visualized along with the registered point cloud.
#         """
#         try:
#             # Initialize a new Open3D Visualizer object.
#             self.vis = o3d.visualization.Visualizer()
#             # Create a visualization window with a specified name.
#             self.vis.create_window(window_name='FlowVis3D')
#             # Add the registered point cloud to the visualizer.
#             self.vis.add_geometry(self.registered_pcd)  # Registered point cloud
#             # Also add the reference geometry that has just been loaded or updated.
#             self.vis.add_geometry(reference_geom)       # Loaded reference geometry
#             # Run the visualizer to display the geometries until the window is closed.
#             self.vis.run()
#             # Once the visualization is done or the window is closed, properly destroy the window to free resources.
#             self.vis.destroy_window()
#         except Exception as e:
#             # If any errors occur during the setup or visualization process, handle them appropriately.
#             self.handleVisualizationError(str(e))

#     def handleVisualizationError(self, error_message):
#         """
#         Handles any errors that may occur during the visualization process by logging the error details.
#         This method can be expanded to include more sophisticated error handling strategies such as user notifications.

#         Args:
#             error_message (str): A message describing the error encountered during visualization.
#         """
#         # Print the error message to the console. This could be replaced or supplemented by more user-visible error reporting mechanisms.
#         print(f"Visualization Error: {error_message}")

#     def initiateSaveData(self):
#         """
#         Initiates the save data process by first verifying that all required fields are filled and checking for existing files.
#         This method ensures that any previous saved data isn't unintentionally overwritten without user consent.
#         """
#         # Validate required fields
#         missing_fields = self.validateRequiredFields()
#         if missing_fields:
#             QMessageBox.warning(self, "Warning", f"Please enter {', '.join(missing_fields)}.")
#             return

#         # Clear any previous messages in the save log.
#         self.saveLogLabel.clear()

#         # Check if a file has been selected to save.
#         if not self.file_path_pcd:
#             QMessageBox.warning(self, "Warning", "No file selected. Please select a file first.")
#             return

#         # Generate the base file name string
#         model = self.modelLineEdit.text()
#         wt_run = self.WTRunLineEdit.text()
#         wt_map = self.WTMapLineEdit.text()
#         car_part = self.carPartComboBox.currentText().strip()
#         load_condition = self.loadConditionComboBox.currentText()
#         base_file_name = f"{model}_{wt_run}_FV_Sauber_STD_MAP_{wt_map}_{car_part}_{load_condition}"

#         # Check for invalid characters in the file name
#         if not self.isValidFileName(base_file_name):
#             QMessageBox.warning(self, "Warning", "Invalid file name. Remove special characters (such as \" or ') from the input fields.")
#             return

#         # Get the directory of the selected file
#         directory = os.path.dirname(self.file_path_pcd)

#         # Check for existing files that contain the base file name
#         existing_files = [f for f in os.listdir(directory) if base_file_name in f]
#         if existing_files:
#             # Prompt the user to confirm overwriting existing files.
#             reply = QMessageBox.question(self, 'Confirm Overwrite',
#                                         f"Files already exist that match the name pattern '{base_file_name}'. Do you want to overwrite them?",
#                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

#             # If the user chooses not to overwrite, cancel the save operation.
#             if reply == QMessageBox.No:
#                 QMessageBox.information(self, "Save Cancelled", "Save operation cancelled by user.")
#                 return

#         # Extract case description from wt_map
#         self.case_description = None
#         case_descriptions = ["SL_01", "LS_02", "LS_03", "VHS_04", "LS_05", "MS_06", "MS_07", "MS_09", "LS_09", "LS_10"]
#         for case in case_descriptions:
#             if case in wt_map:
#                 self.case_description = case
#                 break

#         if self.case_description:
#             case_number = self.case_description.split('_')[1]
#         else:
#             QMessageBox.warning(self, "Warning", "Case description not found in WT Map.")
#             return

#         # Proceed with the save process if the files do not exist or if overwrite is confirmed.
#         self.startSaveProcess(wt_run, self.case_description, case_number)

#     def isValidFileName(self, file_name):
#         """
#         Checks if the provided file name contains any invalid characters.

#         Args:
#             file_name (str): The file name to check.

#         Returns:
#             bool: True if the file name is valid, False otherwise.
#         """
#         invalid_chars = ['"', "'"]
#         for char in invalid_chars:
#             if char in file_name:
#                 return False
#         return True

#     def validateRequiredFields(self):
#         """
#         Validates that all required fields for the file name are filled.

#         Returns:
#             list: A list of missing field names, empty if all fields are filled.
#         """
#         missing_fields = []

#         if not self.modelLineEdit.text():
#             missing_fields.append("Model")
#         if not self.WTRunLineEdit.text():
#             missing_fields.append("WT Run")
#         if not self.WTMapLineEdit.text():
#             missing_fields.append("WT Map")
#         if self.carPartComboBox.currentText() not in self.car_part_correspondences:
#             missing_fields.append("Car Part")
#         if not self.loadConditionComboBox.currentText():
#             missing_fields.append("Load Condition")

#         return missing_fields

#     def startSaveProcess(self, wt_run, case_description, case_number):
#         """
#         Starts the actual save process by setting up a SaveWorker within a new thread.
#         This method manages the lifecycle of the save operation, including starting and cleaning up the thread.
#         """
#         # Check if a save operation is already in progress and warn the user if so.
#         if self.saveThread and self.saveThread.isRunning():
#             QMessageBox.warning(self, "Warning", "A save operation is already in progress. Please wait for it to complete.")
#             return

#         # Get the composed filename from the QLineEdit
#         composed_filename = self.fileNameLineEdit.text()

#         # Instantiate the SaveWorker with the necessary data for saving.
#         self.saveWorker = SaveWorker(self, self.file_path_pcd, self.registered_pcd, self.registered_pcd_scaled, self.registered_mesh, self.registered_mesh_scaled, composed_filename)
#         # Connect the worker's signals to appropriate slots for handling completion, errors, and log messages.
#         self.saveWorker.finished.connect(self.cleanupSaveProcess)
#         self.saveWorker.error.connect(self.handleSaveError)
#         self.saveWorker.log_message.connect(self.updateSaveLog)

#         # Create and configure a new thread for the save operation.
#         self.saveThread = QThread()
#         self.saveWorker.moveToThread(self.saveThread)
#         self.saveThread.started.connect(self.saveWorker.run)
#         self.saveThread.finished.connect(self.saveThread.deleteLater)

#         # Start the save thread and initiate the loading animation to indicate saving is in progress.
#         self.saveThread.start()
#         startLoadingAnimation(self.loadingLabel_savedata, self.loadingMovie_savedata, self.saveSection.scrollArea)

#     def handleSaveError(self, error):
#         """
#         Handles any errors that occur during the save process by displaying an error message to the user and initiating cleanup.

#         Args:
#             error (str): The error message describing what went wrong during the save operation.
#         """
#         # Display a critical error message using a message box to alert the user that an error has occurred.
#         QMessageBox.critical(self, "Save Error", f"An error occurred: {error}")
#         # After displaying the error message, call the cleanup process to properly shut down the worker and thread.
#         self.cleanupSaveProcess()  # Ensures that resources are freed and the application state is correctly reset.

#     def cleanupSaveProcess(self):
#         """
#         Cleans up resources and UI components used during the save process. This method is called after a save operation
#         completes, either successfully or due to an error, to ensure the application returns to a stable state.
#         """
#         # Stop any animations in the UI that indicate ongoing save operations, signaling completion to the user.
#         stopLoadingAnimation(self.loadingLabel_savedata, self.loadingMovie_savedata, self.saveSection.scrollArea)

#         # Check if there is an active save worker and ensure it is properly deleted to free resources.
#         if self.saveWorker:
#             self.saveWorker.deleteLater()  # Safely delete the worker object.
#             self.saveWorker = None         # Remove the reference to the worker.

#         # Check if there is an active save thread and ensure it is properly terminated.
#         if self.saveThread:
#             self.saveThread.quit()        # Request the thread to quit.
#             self.saveThread.wait()        # Wait for the thread to finish.
#             self.saveThread.deleteLater() # Safely delete the thread object.
#             self.saveThread = None

#         # Uncomment below if you wish to provide user feedback when save operations are definitively completed.
#         # QMessageBox.information(self, "Save Complete", "Data has been successfully saved.")

#     def updateSaveLog(self, message):
#         """
#         Updates the save log text edit widget with messages received from the SaveWorker.
#         This function is typically used to display status updates and results from the save process.

#         Args:
#             message (str): The message to be added to the save log.
#         """
#         # Append the provided message to the save log QTextEdit widget.
#         # This allows users to see the progress and details of the save operation in real-time.
#         self.saveLogLabel.append(message)

#     def appendLog(self, message):
#         """
#         Appends a given message to the save log. This method can be used for general logging
#         outside of the specific save operation context.

#         Args:
#             message (str): The message to append to the log.
#         """
#         # Append the provided message to the save log QTextEdit widget.
#         # This method is a general utility for logging various types of messages.
#         self.saveLogLabel.append(message)  # Append message to the QTextEdit

#     def uploadtoSandbox(self):
#         """
#         Initiates the upload of registered data to a sandbox environment.
#         """
#         # Clear any previous logs in the sandbox log label.
#         self.sandboxLogLabel.clear()

#         # Re-initialize the mesh_upload variable to ensure it is updated
#         self.mesh_upload = None

#         # Retrieve user inputs from the UI.
#         model = self.modelLineEdit.text()
#         wt_run = self.WTRunLineEdit.text().strip()
#         car_part = self.carPartComboBox.currentText().strip()
#         wt_map = self.WTMapLineEdit.text().strip()

#         # Ensure necessary inputs are provided.
#         if not self.case_description:
#             QMessageBox.warning(self, "Warning", "Please enter the case description in the WT Map.")
#             return
#         if not wt_run:
#             QMessageBox.warning(self, "Warning", "Please enter WT Run.")
#             return
#         if not model:
#             QMessageBox.warning(self, "Warning", "Please enter WT Model.")
#             return

#         # Prevent multiple concurrent upload operations.
#         if self.uploadThread and self.uploadThread.isRunning():
#             QMessageBox.warning(self, "Warning", "An upload operation is already in progress. Please wait for it to complete.")
#             return

#         # Show the upload options dialog
#         dialog = UploadOptionsDialog(self)
#         if dialog.exec() == QDialog.Accepted:
#             upload_type, scale = dialog.getSelectedOptions()

#             if upload_type == "registered":
#                 if self.registered_mesh is not None:
#                     self.mesh_upload = self.registered_mesh
#                     self.continueUploadProcess()
#                 else:
#                     QMessageBox.warning(self, "Warning", "No registered mesh available.")
#             elif upload_type == "input":
#                 if self.file_path_pcd is not None:
#                     if scale is None:
#                         QMessageBox.warning(self, "Warning", "Please select the input file scale.")
#                         return
#                     # Start the loading animation to indicate an ongoing process.
#                     startLoadingAnimation(self.loadingLabel_upload, self.loadingMovie_upload, self.saveSection.scrollArea)

#                     # Create and start the mesh loading worker.
#                     self.meshWorker = LoadMeshWorker(self.file_path_pcd, scale)
#                     self.meshThread = QThread()
#                     self.meshWorker.moveToThread(self.meshThread)
#                     self.meshWorker.finished.connect(self.handleMeshLoadComplete)
#                     self.meshWorker.finished.connect(self.meshThread.quit)
#                     self.meshWorker.finished.connect(self.meshWorker.deleteLater)
#                     self.meshThread.finished.connect(self.meshThread.deleteLater)
#                     self.meshThread.started.connect(self.meshWorker.run)
#                     self.meshThread.start()
#                 else:
#                     QMessageBox.warning(self, "Warning", "No input mesh available.")
#         else:
#             # User cancelled the dialog
#             return

#     def handleMeshLoadComplete(self, mesh, error_message):
#         """
#         Handles the completion of the mesh loading process.
#         """
#         # Stop the loading animation regardless of success or error.
#         stopLoadingAnimation(self.loadingLabel_upload, self.loadingMovie_upload, self.saveSection.scrollArea)

#         # Handle errors during mesh loading.
#         if error_message:
#             QMessageBox.critical(self, "Mesh Load Error", error_message)
#             return

#         # If a mesh was successfully loaded, proceed with the upload process.
#         if mesh:
#             self.mesh_upload = mesh
#             self.continueUploadProcess()
#         else:
#             # Alert the user if no mesh was loaded.
#             QMessageBox.warning(self, "Load Warning", "The input file has no mesh data.")

#     def continueUploadProcess(self):
#         """
#         Continues the upload process after verifying all necessary data is available.
#         """
#         # Extract the map conversion name from the WT map.
#         wt_map = self.WTMapLineEdit.text().strip()
#         map_conversion_name = self.extractMapConversionName(wt_map)
#         if not map_conversion_name:
#             QMessageBox.warning(self, "Warning", "Please enter the WT map conversion name.")
#             return

#         # Define the target upload directory.
#         self.target_directory = r"//srvnetapp00/Technical/Aerodynamics/Development/SANDBOX"

#         # Extract the case number from the case description.
#         case_number = self.case_description.split('_')[1]

#         # Retrieve user inputs from the UI.
#         model = self.modelLineEdit.text()
#         wt_run = self.WTRunLineEdit.text().strip()
#         car_part = self.carPartComboBox.currentText().strip()

#         # Instantiate the UploadWorker with the necessary parameters.
#         self.uploadWorker = UploadWorker(self, self.mesh_upload, car_part, self.target_directory, wt_run, self.case_description, map_conversion_name, case_number)
        
#         # Connect the worker's signals to appropriate slots.
#         self.uploadWorker.finished.connect(self.cleanupUploadProcess)
#         self.uploadWorker.error.connect(self.handleUploadError)
#         self.uploadWorker.log_message.connect(self.updateUploadLog)

#         # Create and start the thread for the upload process.
#         self.uploadThread = QThread()
#         self.uploadWorker.moveToThread(self.uploadThread)
#         self.uploadThread.started.connect(self.uploadWorker.run)
#         self.uploadThread.finished.connect(self.uploadThread.deleteLater)
#         self.uploadThread.start()

#         # Start the loading animation to indicate an ongoing upload process.
#         startLoadingAnimation(self.loadingLabel_upload, self.loadingMovie_upload, self.saveSection.scrollArea)

#     def updateScaleSelection(self):
#         """
#         Updates the scale selection based on the checkbox state.
#         Ensures only one checkbox is selected at a time.
#         """
#         if self.sender() == self.wt_checkbox:
#             if self.wt_checkbox.isChecked():
#                 # Uncheck the other checkbox and set the selection to WT.
#                 self.cfd_checkbox.setChecked(False)
#                 self.scale_selection = "WT"
#             else:
#                 self.scale_selection = None
#         elif self.sender() == self.cfd_checkbox:
#             if self.cfd_checkbox.isChecked():
#                 # Uncheck the other checkbox and set the selection to CFD.
#                 self.wt_checkbox.setChecked(False)
#                 self.scale_selection = "CFD"
#             else:
#                 self.scale_selection = None

#     def extractMapConversionName(self, wt_map):
#         """
#         Extracts the map conversion name from the WT map string.

#         Args:
#             wt_map (str): The WT map string.

#         Returns:
#             str: The extracted map conversion name, or None if not found.
#         """
#         parts = wt_map.split('_')
#         if len(parts) >= 3:
#             return parts[2]
#         return None
    
#     def cleanupUploadProcess(self):
#         """
#         Cleans up resources and UI components used during the upload process.
#         This method is called after an upload operation completes, either successfully or due to an error,
#         to ensure the application returns to a stable state.
#         """
#         # Stop the upload animation to indicate that the upload operation is complete.
#         stopLoadingAnimation(self.loadingLabel_upload, self.loadingMovie_upload, self.saveSection.scrollArea)

#         # Check if there is an active upload worker and ensure it is properly deleted to free resources.
#         if self.uploadWorker:
#             self.uploadWorker.deleteLater()  # Safely delete the worker object.
#             self.uploadWorker = None         # Remove the reference to the worker.

#         # Check if there is an active upload thread and ensure it is properly terminated.
#         if self.uploadThread:
#             self.uploadThread.quit()        # Request the thread to quit.
#             self.uploadThread.wait()        # Wait for the thread to finish.
#             self.uploadThread.deleteLater() # Safely delete the thread object.
#             self.uploadThread = None

#         # Uncomment below if you wish to provide user feedback when upload operations are definitively completed.
#         # QMessageBox.information(self, "Upload Complete", "Data has been successfully uploaded.")

#     def handleUploadError(self, error):
#         """
#         Handles any errors that occur during the upload process by displaying an error message to the user and initiating cleanup.

#         Args:
#             error (str): The error message describing what went wrong during the upload operation.
#         """
#         # Display a critical error message using a message box to alert the user that an error has occurred.
#         QMessageBox.critical(self, "Upload Error", f"An error occurred: {error}")
#         # After displaying the error message, call the cleanup process to properly shut down the worker and thread.
#         self.cleanupUploadProcess()  # Ensures that resources are freed and the application state is correctly reset.

#     def updateUploadLog(self, message):
#         """
#         Updates the upload log text edit widget with messages received from the UploadWorker.
#         This function is typically used to display status updates and results from the upload process.

#         Args:
#             message (str): The message to be added to the upload log.
#         """
#         # Append the provided message to the upload log QTextEdit widget.
#         # This allows users to see the progress and details of the upload operation in real-time.
#         self.sandboxLogLabel.append(message)


# class UploadOptionsDialog(QDialog):
#     """
#     Dialog to gather upload options from the user.
#     """
#     def __init__(self, parent=None):
#         """
#         Initializes the UploadOptionsDialog with upload options and scales.
#         """
#         super().__init__(parent)
#         self.setWindowTitle("Upload to Sandbox")
#         self.setModal(True)

#         # Create the main layout
#         self.layout = QVBoxLayout(self)

#         # Create a QButtonGroup for mutual exclusivity
#         self.uploadOptionGroup = QButtonGroup(self)

#         # Create a horizontal layout for the main options
#         main_options_layout = QHBoxLayout()

#         # Upload options
#         self.registered_mesh_radio = QRadioButton("Registered Mesh")
#         self.input_mesh_radio = QRadioButton("Input Mesh (if available)")

#         # Add radio buttons to the button group
#         self.uploadOptionGroup.addButton(self.registered_mesh_radio)
#         self.uploadOptionGroup.addButton(self.input_mesh_radio)

#         main_options_layout.addWidget(self.registered_mesh_radio)
#         main_options_layout.addWidget(self.input_mesh_radio)

#         self.layout.addLayout(main_options_layout)

#         # Create a grid layout to position elements correctly
#         grid_layout = QGridLayout()

#         # Add registered mesh option to the left column
#         grid_layout.addWidget(self.registered_mesh_radio, 0, 0)

#         # Add input mesh option to the right column
#         grid_layout.addWidget(self.input_mesh_radio, 0, 1)

#         # Create a vertical layout for the input mesh options in the right column
#         self.input_mesh_options_layout = QVBoxLayout()

#         # Add the always visible text
#         self.scale_question_label = QLabel("What is the input mesh scale?")
#         self.input_mesh_options_layout.addWidget(self.scale_question_label)

#         # Add the WT/CFD scale options
#         self.wt_scale_checkbox = QCheckBox("WT Scale (60%)")
#         self.cfd_scale_checkbox = QCheckBox("CFD Scale (100%)")
#         self.wt_scale_checkbox.setDisabled(True)
#         self.cfd_scale_checkbox.setDisabled(True)

#         self.input_mesh_options_layout.addWidget(self.wt_scale_checkbox)
#         self.input_mesh_options_layout.addWidget(self.cfd_scale_checkbox)

#         # Add input mesh options to the grid layout in the right column
#         grid_layout.addLayout(self.input_mesh_options_layout, 1, 1)

#         # Add the grid layout to the main layout
#         self.layout.addLayout(grid_layout)

#         # Connect input mesh radio button to enable/disable scale options
#         self.input_mesh_radio.toggled.connect(self.toggleScaleOptions)

#         # Dialog buttons
#         self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

#         # Retrieve OK and Cancel buttons from button_box
#         ok_button = self.button_box.button(QDialogButtonBox.Ok)
#         cancel_button = self.button_box.button(QDialogButtonBox.Cancel)

#         # Create a horizontal layout for the buttons
#         button_layout = QHBoxLayout()
#         button_layout.addWidget(ok_button)
#         button_layout.addStretch()
#         button_layout.addWidget(cancel_button)
        
#         self.layout.addLayout(button_layout)

#         # Connect the buttons to dialog accept and reject methods
#         self.button_box.accepted.connect(self.verifySelection)
#         self.button_box.rejected.connect(self.reject)

#     def toggleScaleOptions(self, checked):
#         """
#         Enables or disables the scale options based on the input mesh selection.
#         """
#         self.wt_scale_checkbox.setDisabled(not checked)
#         self.cfd_scale_checkbox.setDisabled(not checked)

#     def verifySelection(self):
#         """
#         Verifies the selection before accepting the dialog.
#         """
#         if self.registered_mesh_radio.isChecked():
#             self.accept()
#         elif self.input_mesh_radio.isChecked():
#             if self.wt_scale_checkbox.isChecked() or self.cfd_scale_checkbox.isChecked():
#                 self.accept()
#             else:
#                 QMessageBox.warning(self, "Warning", "Please select the input file scale.")
#         else:
#             QMessageBox.warning(self, "Warning", "Please select an upload option.")

#     def getSelectedOptions(self):
#         """
#         Retrieves the selected upload options.

#         Returns:
#             tuple: A tuple containing the upload type ("registered" or "input") and the scale ("WT" or "CFD") if applicable.
#         """
#         if self.registered_mesh_radio.isChecked():
#             return "registered", None
#         elif self.input_mesh_radio.isChecked():
#             scale = "WT" if self.wt_scale_checkbox.isChecked() else "CFD" if self.cfd_scale_checkbox.isChecked() else None
#             return "input", scale
#         return None, None



# # Run the script as the main program.
# if __name__ == "__main__":
#     # Initialize the QApplication object
#     app = QApplication([])  # The list can hold command line arguments passed to the application.

#     # Set a dark theme for the entire application by modifying the style sheet of the application.
#     # This changes the background color to black and the text color to white for all QWidget objects.
#     app.setStyleSheet("""
#         QWidget {
#             background-color: black;
#             color: white;
#         }
#     """)

#     # Set a custom font for the entire application.
#     nunitoFont = QFont("Nunito", 10)  
#     app.setFont(nunitoFont)  # Apply the font to the application, affecting all text in the UI.

#     # Create an instance of the MainApp class. 
#     window = MainApp()

#     # Execute the application's main loop
#     app.exec()
