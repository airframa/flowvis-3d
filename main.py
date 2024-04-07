from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLineEdit, QFileDialog, QLabel, QTextEdit
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtCore import Qt, QTimer
from components.point_cloud_utils import load_pcd
from components.point_cloud_registration import PointCloudRegistration
import open3d as o3d

print('Yah mon, starting up...')  # Should print immediately

# Define a QWidget subclass that supports drag-and-drop functionality.
class DroppableWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Enable the widget to accept dropped data.
        self.setAcceptDrops(True)

    # Override the dragEnterEvent to respond to drag actions.
    def dragEnterEvent(self, event):
        # Check if the dragged data contains URLs (file paths).
        if event.mimeData().hasUrls():
            # Accept the action proposed by the drag event.
            event.acceptProposedAction()

    # Override the dropEvent to handle the actual drop action.
    def dropEvent(self, event):
        # Extract file paths from the dropped URLs.
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        # If there are files, update the file path line edit with the first file path.
        if files:
            self.parent().filePathLineEdit.setText(files[0])

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

        centralWidget = DroppableWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)

        # Load and display the logo at the top
        self.logoLabel = QLabel()
        self.logoPixmap = QPixmap("./ui/FlowVis3D_logo_v2.jpg")
        self.logoPixmap = self.logoPixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logoLabel.setPixmap(self.logoPixmap)
        self.logoLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.logoLabel)

        # Visualization Section
        visualizationSection = CollapsibleSection("Load", self)
        layout.addWidget(visualizationSection)

        # Load section explanation text, adjusted styling for padding
        loadExplanation = QLabel("Select a .ply or .stl file and display the point cloud.")
        loadExplanation.setWordWrap(True)
        loadExplanation.setStyleSheet("color: white; padding-top: -10px; padding-bottom: 10px;")  # Adjusted padding
        visualizationSection.contentLayout().addWidget(loadExplanation)
        # Add spacing after the explanation text for more separation
        visualizationSection.contentLayout().addSpacing(10)

        # Add widgets to the visualization section
        # Select file
        self.selectFileButton = QPushButton("Select Point Cloud")
        self.selectFileButton.setStyleSheet("QPushButton { background-color: red; color: white; font-weight: bold; }")
        self.selectFileButton.clicked.connect(self.openFileDialog)
        visualizationSection.contentLayout().addWidget(self.selectFileButton)
        self.filePathLineEdit = QLineEdit()
        visualizationSection.contentLayout().addWidget(self.filePathLineEdit)

        # Show 3D Point Cloud
        self.visualizeButton = QPushButton("Visualize Point Cloud")
        self.visualizeButton.setStyleSheet("QPushButton { background-color: red; color: white; font-weight: bold; }")
        self.visualizeButton.clicked.connect(self.visualizePointCloud)
        visualizationSection.contentLayout().addWidget(self.visualizeButton)

        # Registration Section with explanation and new UI elements
        registrationSection = CollapsibleSection("Register", self)
        layout.addWidget(registrationSection)

        # Registration section explanation text
        registerExplanation = QLabel("Select a reference .stl file and register the point cloud to the CFD or WT coordinates.")
        registerExplanation.setWordWrap(True)
        registerExplanation.setStyleSheet("color: white; padding-top: -5px; padding-bottom: 10px;")
        registrationSection.contentLayout().addWidget(registerExplanation)
        registrationSection.contentLayout().addSpacing(10)

        # "Select Reference" button
        self.selectReferenceButton = QPushButton("Select Reference")
        self.selectReferenceButton.setStyleSheet("QPushButton { background-color: red; color: white; font-weight: bold; }")
        self.selectReferenceButton.clicked.connect(self.selectReferenceFileDialog)
        registrationSection.contentLayout().addWidget(self.selectReferenceButton)

        # Drag and Drop item for selecting reference file (referenceFilePathLineEdit)
        self.referenceFilePathLineEdit = QLineEdit()
        registrationSection.contentLayout().addWidget(self.referenceFilePathLineEdit)

        # Registration Section enhanced with an "Execute Registration" button
        self.executeRegistrationButton = QPushButton("Register")
        self.executeRegistrationButton.setStyleSheet("QPushButton { background-color: red; color: white; font-weight: bold; }")
        self.executeRegistrationButton.clicked.connect(self.executeRegistration)
        registrationSection.contentLayout().addWidget(self.executeRegistrationButton)

        # Add a QLabel for displaying registration logs
        self.registrationLogLabel = QLabel()
        self.registrationLogLabel.setWordWrap(True)
        # self.registrationLogLabel.setStyleSheet("""
        #     QLabel {
        #         color: white;
        #         background-color: black;
        #         padding-top: 10px;
        #         padding-bottom: 10px;
        #         margin-top: 10px;  # Adds some space between the button and the label
        #     }
        # """)
        self.registrationLogLabel.setStyleSheet("""
            QLabel {
                color: white;
                background-color: black;
                padding-top: 5px;
                padding-bottom: 5px;
                margin-top: 5px; /* Adds some space between the button and the label */
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

    # Define the method to open a file dialog and update the file path line edit.
    def openFileDialog(self):
        # Specify the initial directory for the file dialog
        initialDir = "\\\\srvnetapp00\\Technical\\Aerodynamics\\Development\\FlowViz"
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Point Cloud File", initialDir, "Point Cloud Files (*.ply *.stl)")
        if file_name:
            self.filePathLineEdit.setText(file_name)

    def selectReferenceFileDialog(self):
        # Method to open a file dialog for selecting the reference file
        initialDir = "\\\\srvnetapp00\\Technical\\Aerodynamics\\Development\\FlowViz"
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Reference File", initialDir, "Reference Files (*.ply)")
        if file_name:
            self.referenceFilePathLineEdit.setText(file_name)

    # Define the method to load and visualize the point cloud using Open3D.
    def visualizePointCloud(self):
        file_path = self.filePathLineEdit.text()
        if file_path:
            pcd = load_pcd(file_path)  # Load the point cloud data.
            o3d.visualization.draw_geometries([pcd])  # Visualize the point cloud.
        else:
            print("No file selected.")

    def executeRegistration(self):
        sourceFilePath = self.filePathLineEdit.text()
        referenceFilePath = self.referenceFilePathLineEdit.text()

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


