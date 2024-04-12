from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLineEdit, QFileDialog, QLabel, QTextEdit, QStackedWidget
from PySide6.QtGui import QFont, QPixmap, QDragEnterEvent, QDropEvent, QIcon
from PySide6.QtCore import Qt, QSize
from components.point_cloud_utils import load_pcd
from components.point_cloud_registration import PointCloudRegistration
import open3d as o3d

print('Yah mon, starting up...')  # Should print immediately

class IconTextButton(QWidget):
    def __init__(self, iconPath, text, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.iconLabel = QLabel()
        self.iconLabel.setPixmap(QPixmap(iconPath).scaled(60, 60, Qt.KeepAspectRatio))
        self.textLabel = QLabel(text)
        
        # Customize label styles as needed
        self.textLabel.setAlignment(Qt.AlignCenter)
        self.iconLabel.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.iconLabel)
        layout.addWidget(self.textLabel)
        
        # Set the overall layout and style
        self.setLayout(layout)
        self.setStyleSheet("background-color: none;")

        # Make this widget act like a button
        self.pushButton = QPushButton(self)
        self.pushButton.setStyleSheet("background-color: transparent;")
        self.pushButton.setCursor(Qt.PointingHandCursor)
        self.pushButton.clicked.connect(self.onClick)
        self.pushButton.resize(self.sizeHint())

    def onClick(self):
        print(f"Clicked: {self.textLabel.text()}")
        # Emit signal or call function here

class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setWindowTitle("FlowVis3D")
        # self.setMinimumSize(800, 600)  # Set a minimum size for the window
        self.setupUI()

    def setupUI(self):
        # Create a central widget
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        # Main layout is vertical
        mainLayout = QVBoxLayout(centralWidget)

        # Logo at the top
        self.logoLabel = QLabel()
        self.logoPixmap = QPixmap("./ui/FlowVis3D_logo_v2.jpg").scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logoLabel.setPixmap(self.logoPixmap)
        self.logoLabel.setAlignment(Qt.AlignCenter)
        mainLayout.addWidget(self.logoLabel)

        # Create a horizontal layout for navigation panel and content
        contentLayout = QHBoxLayout()

        # Navigation panel on the left
        navigationPanel = QWidget()
        navigationLayout = QVBoxLayout(navigationPanel)
        navigationLayout.setAlignment(Qt.AlignTop)  # Align the navigation buttons to the top
        navigationPanel.setFixedWidth(200)  # Adjust width as needed


        # Style options for navigation buttons
        buttonStyle = "QPushButton { font-weight: bold; font-size: 16px; }"

        # Initialize buttons
        self.loadPageButton = QPushButton(" Visualize")  # Space for alignment if needed
        self.registerPageButton = QPushButton(" Register")

        # Load icons
        loadIcon = QPixmap("./ui/load_pcd_icon_v3.jpg")
        registerIcon = QPixmap("./ui/register_pcd_icon_v1.jpg")

        # Set icons on buttons
        self.loadPageButton.setIcon(QIcon(loadIcon))
        self.registerPageButton.setIcon(QIcon(registerIcon))

        # Optionally adjust the icon size if the defaults are not suitable
        self.loadPageButton.setIconSize(QSize(125, 125))
        self.registerPageButton.setIconSize(QSize(90, 90))
        buttonStyle = """
        QPushButton {
            font-weight: bold;
            font-size: 16px;
            padding-top: 20px; /* Adjust these paddings */
            padding-bottom: 20px; /* Adjust these paddings */
            text-align: left;
            icon-size: 60px, 60px; /* Adjust icon size here */
        }
        """
        self.loadPageButton.setStyleSheet(buttonStyle)
        self.registerPageButton.setStyleSheet(buttonStyle)

        # Apply styles and connect signals
        buttonStyle = "QPushButton { font-weight: bold; font-size: 16px; }"
        self.loadPageButton.setStyleSheet(buttonStyle)
        self.registerPageButton.setStyleSheet(buttonStyle)
        self.loadPageButton.clicked.connect(self.showLoadPage)
        self.registerPageButton.clicked.connect(self.showRegisterPage)

        # Add buttons to the navigation layout
        navigationLayout.addWidget(self.loadPageButton)
        navigationLayout.addWidget(self.registerPageButton)

        # Add the navigation panel to the content layout
        contentLayout.addWidget(navigationPanel, 1)  # The second parameter is the stretch factor

        # Stacked widget for the pages, added to the content layout with a larger stretch factor
        self.stackedWidget = QStackedWidget()
        contentLayout.addWidget(self.stackedWidget, 3)  # Giving more space to the content area

        # Add the content layout below the logo
        mainLayout.addLayout(contentLayout)

        # Adding pages to the stacked widget
        self.loadPage = LoadPage()
        self.registrationPage = RegistrationPage()
        self.stackedWidget.addWidget(self.loadPage)
        self.stackedWidget.addWidget(self.registrationPage)

    def showLoadPage(self):
        self.stackedWidget.setCurrentWidget(self.loadPage)

    def showRegisterPage(self):
        self.stackedWidget.setCurrentWidget(self.registrationPage)


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
            self.loadFileLineEdit.setText(path)

    def getFilePath(self):
        return self.fileLineEdit.text()

    def setLoadFilePath(self, file_path):
        self.filePathLineEdit.setText(file_path)

    def setReferenceFilePath(self, file_path):
        self.referenceFilePathLineEdit.setText(file_path)

class LoadPage(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            # Add your load functionality widgets here
            layout.addWidget(QLabel("Load your point cloud files here."))

class RegistrationPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        # Add your registration functionality widgets here
        layout.addWidget(QLabel("Register your point cloud files here."))


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
    msFont = QFont("Montserrat", 10)  # Adjust the size as needed
    app.setFont(msFont)

    window = MainApp()
    window.show()
    app.exec()


