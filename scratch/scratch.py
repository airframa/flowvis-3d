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