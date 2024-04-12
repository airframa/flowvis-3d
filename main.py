from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLineEdit, QFileDialog, QLabel, QTextEdit
from PySide6.QtGui import QFont, QPixmap, QDragEnterEvent, QDropEvent
from PySide6.QtCore import Qt
from components.point_cloud_utils import load_pcd
from components.point_cloud_registration import PointCloudRegistration
import open3d as o3d

print('Yah mon, starting up...')  # Should print immediately


class FileChooserWidget(QWidget):
    def __init__(self, button_text="Select File", initial_dir="", parent=None):
        super().__init__(parent)
        self.initial_dir = initial_dir
        self.button_text = button_text
        self.setupUI()

    def setupUI(self):
        self.layout = QVBoxLayout(self)

        # Select File Button with styled appearance
        self.selectFileButton = QPushButton(self.button_text)
        self.selectFileButton.setStyleSheet("QPushButton { background-color: red; color: white; font-weight: bold; }")
        self.selectFileButton.setMinimumHeight(40)  # Set minimum height to make button larger
        self.layout.addWidget(self.selectFileButton)
        self.selectFileButton.clicked.connect(self.openFileDialog)

        # File path line edit, acting as the drag-and-drop area
        self.fileLineEdit = QLineEdit()
        self.fileLineEdit.setReadOnly(True)
        self.fileLineEdit.setMinimumHeight(30)  # Slightly larger line edit for better visual
        self.layout.addWidget(self.fileLineEdit)

        self.setAcceptDrops(True)

    def openFileDialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, self.button_text, self.initial_dir, "Point Cloud Files (*.ply *.stl);;All Files (*)")
        if file_name:
            self.fileLineEdit.setText(file_name)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.fileLineEdit.setText(path)

    def getFilePath(self):
        """Return the currently selected file path."""
        return self.fileLineEdit.text()


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

class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setWindowTitle("FlowVis3D")
        self.setupUI()

    def setupUI(self):
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)

        # Load and display the logo at the top
        self.logoLabel = QLabel()
        self.logoPixmap = QPixmap("./ui/FlowVis3D_logo_v2.jpg").scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logoLabel.setPixmap(self.logoPixmap)
        self.logoLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.logoLabel)

        # Visualization Section with CollapsibleSection
        visualizationSection = CollapsibleSection("Load", self)
        layout.addWidget(visualizationSection)

        # Load section with FileChooserWidget
        self.loadFileChooser = FileChooserWidget("Select Point Cloud", "\\\\srvnetapp00\\Technical\\Aerodynamics\\Development\\FlowViz", self)
        visualizationSection.contentLayout().addWidget(self.loadFileChooser)

        # Show 3D Point Cloud Button within the visualization section
        self.visualizeButton = QPushButton("Visualize Point Cloud")
        self.visualizeButton.setStyleSheet("QPushButton { background-color: red; color: white; font-weight: bold; }")
        self.visualizeButton.clicked.connect(self.visualizePointCloud)
        visualizationSection.contentLayout().addWidget(self.visualizeButton)

        # Register Section with CollapsibleSection
        registrationSection = CollapsibleSection("Register", self)
        layout.addWidget(registrationSection)

        # Register FileChooserWidget for "Register" section
        self.registerFileChooser = FileChooserWidget("Select Reference", "\\\\srvnetapp00\\Technical\\Aerodynamics\\Development\\FlowViz", self)
        registrationSection.contentLayout().addWidget(self.registerFileChooser)

        # Execute Registration Button within the registration section
        self.executeRegistrationButton = QPushButton("Register Point Cloud")
        self.executeRegistrationButton.setStyleSheet("QPushButton { background-color: red; color: white; font-weight: bold; }")
        self.executeRegistrationButton.clicked.connect(self.executeRegistration)
        registrationSection.contentLayout().addWidget(self.executeRegistrationButton)

        # Registration logs display
        self.registrationLogLabel = QLabel()
        self.registrationLogLabel.setWordWrap(True)
        self.registrationLogLabel.setStyleSheet("""
            QLabel {
                color: white;
                background-color: black;
                padding-top: 5px;
                padding-bottom: 5px;
                margin-top: 5px;
            }
        """)
        registrationSection.contentLayout().addWidget(self.registrationLogLabel)

        # Add spacing after the new elements for more separation
        # registrationSection.contentLayout().addSpacing(10)

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

    def setLoadFilePath(self, file_path):
        self.filePathLineEdit.setText(file_path)

    def setReferenceFilePath(self, file_path):
        self.referenceFilePathLineEdit.setText(file_path)

    # Define the method to load and visualize the point cloud using Open3D.
    def visualizePointCloud(self):
        file_path = self.loadFileChooser.getFilePath()  # Use the getFilePath method from FileChooserWidget
        if file_path:
            pcd = load_pcd(file_path)  # Load the point cloud data.
            o3d.visualization.draw_geometries([pcd])  # Visualize the point cloud.
        else:
            print("No file selected.")

    def executeRegistration(self):
        sourceFilePath = self.loadFileChooser.getFilePath()  # Use the getFilePath method from FileChooserWidget
        referenceFilePath = self.registerFileChooser.getFilePath()  # Use the getFilePath method from FileChooserWidget

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

        # Display the log text in the QLabel
        self.registrationLogLabel.setText(log_text)
        
        # Process the registered point cloud as needed
        print("Registration completed.")
        # For example, update the visualization or inform the user of the completion

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


