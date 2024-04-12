from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLineEdit, QFileDialog, QLabel, QTextEdit
from PySide6.QtGui import QFont, QPixmap, QDragEnterEvent, QDropEvent, QTextOption, QGuiApplication
from PySide6.QtCore import Qt
from components.point_cloud_utils import load_pcd
from components.point_cloud_registration import PointCloudRegistration
import open3d as o3d

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
        self.setAcceptDrops(True)  # Enable drag and drop for the main window
        self.setupUI()

    def setupUI(self):
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)

        # Load and display the logo at the top
        self.logoLabel = QLabel()
        self.logoPixmap = QPixmap("./ui/FlowVis3D_logo_v2.jpg").scaled(575, 575, Qt.KeepAspectRatio, Qt.SmoothTransformation)
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

        # # Preprocessing Section (placeholder)
        # preprocessSection = CollapsibleSection("Pre-process", self)
        # layout.addWidget(preprocessSection)
        # # Populate preprocessSection.contentLayout() as needed in the future

        # # Segmentation Section (placeholder)
        # segmentSection = CollapsibleSection("Segment", self)
        # layout.addWidget(segmentSection)
        # # Populate segmentSection.contentLayout() as needed in the future

        # # Segmentation Section (placeholder)
        # rasterSection = CollapsibleSection("Raster", self)
        # layout.addWidget(rasterSection)
        # # Populate rasterSection.contentLayout() as needed in the future

        self.show()

    def openFileDialog(self, lineEdit, initial_dir):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Point Cloud File", initial_dir, "Point Cloud Files (*.ply *.stl);;All Files (*)")
        if file_name:
            lineEdit.setText(file_name)

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
        sourceFilePath = self.loadFileLineEdit.text()  # Use the getFilePath method from FileChooserWidget
        referenceFilePath = self.loadRefFileLineEdit.text()  # Use the getFilePath method from FileChooserWidget

        # Ensure both source and reference paths are provided
        if not sourceFilePath or not referenceFilePath:
            self.registrationLogLabel.setText("Source or reference file path is missing.")
            return

        # Load the source point cloud and reference model
        sourcePcd = load_pcd(sourceFilePath)
        # Assuming load_pcd works for your reference file format; otherwise, adjust accordingly

        # Initialize the registration process
        registration = PointCloudRegistration(source=sourcePcd, target=referenceFilePath)
        
        # Perform registration with predefined parameters
        # Assuming `registration.register()` now also returns a log text
        pcd_registered, transformation, log_text = registration.register(desired_fitness_ransac=0.85, desired_fitness_icp=[0.65, 0.75, 0.85])

        # Example: After calculating the transformation matrix
        self.transformationMatrixText = str(transformation)
        self.registrationLogLabel.append("Registration completed. Transformation Matrix:\n" + self.transformationMatrixText)

        # Display the log text in the QLabel
        self.registrationLogLabel.setText(log_text)
        
        # Process the registered point cloud as needed
        print("Registration completed.")
        # For example, update the visualization or inform the user of the completion

    def copyTransformationToClipboard(self):
        clipboard = QGuiApplication.clipboard()
        transformation_text = self.transformationMatrixText  # Ensure this contains the transformation matrix text
        clipboard.setText(transformation_text)
        print("Transformation matrix copied to clipboard.")

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


