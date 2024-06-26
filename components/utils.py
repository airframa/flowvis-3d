from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLineEdit, QLabel, QScrollArea, QToolButton, QStyledItemDelegate
from PySide6.QtGui import  QMovie, QIcon
from PySide6.QtCore import Qt, QSize

class CollapsibleSection(QWidget):
    """
    A custom QWidget that provides a collapsible section within a GUI.
    This section includes a title bar with a toggle button to show/hide the content and an information button.
    It is designed to be embedded within parent widgets and can contain any number of child widgets arranged vertically.
    
    Attributes:
        title (str): The title displayed on the collapsible section header.
        parent (optional QWidget): The parent widget to which this widget belongs.
        height_threshold (int): Not used in this implementation. Can be used to set a height limit for the collapsible behavior.
        info_text (str): Text that will be displayed as a tooltip on the info button.
    """

    def __init__(self, title, parent=None, height_threshold=350, info_text=""):
        """
        Initializes a new instance of the CollapsibleSection with a title, optional parent, height threshold, and information text.
        
        Parameters:
            title (str): The title for the section.
            parent (QWidget, optional): The parent widget.
            height_threshold (int, optional): The maximum height at which the section remains expanded by default.
            info_text (str, optional): Tooltip text for the information button.
        """
        super().__init__(parent)

        # Set height threshold
        # Obtain the primary screen's resolution to determine appropriate maximum section height
        screen = QApplication.primaryScreen()
        size = screen.size()
        # Choose the size of the pixmap based on the width of the screen.
        if size.width() <= 1550:  # Assuming 1550 is typical for smaller, laptop screens
            self.height_threshold = 225               # Maximum height before the collapsible behavior is triggered: set fixed value in case of small screens
        else:
            self.height_threshold = height_threshold  # Maximum height before the collapsible behavior is triggered: set std value

        # Main layout of the widget, using vertical box layout.
        self.layout = QVBoxLayout(self)
        self.setStyleSheet("background-color: black;")  # Set the widget's background color to black.

        # Header layout setup using horizontal box layout.
        self.header = QHBoxLayout()
        self.titleLabel = QLabel(title)  # Label to display the title.
        self.titleLabel.setStyleSheet("font-weight: bold; color: white;")  # Title styling.
        self.header.addWidget(self.titleLabel)  # Adding the title label to the header layout.

        # Toggle button setup to show/hide the collapsible content.
        self.toggleButton = QPushButton("+")
        self.toggleButton.setFixedSize(30, 30)  # Size of the toggle button.
        self.toggleButton.setStyleSheet("font-size: 18px; font-weight: bold; color: white; background-color: black;")
        self.toggleButton.setCheckable(True)
        self.toggleButton.setChecked(False)  # Start with the content hidden.
        self.toggleButton.clicked.connect(self.onToggle)  # Connect the toggle button click event to the onToggle method.
        self.header.addWidget(self.toggleButton)  # Adding the toggle button to the header layout.

        # Information button setup with a tooltip.
        self.infoButton = QToolButton()
        self.infoButton.setIcon(QIcon("./ui/info_icon_v2.png"))  # Set an information icon.
        self.infoButton.setStyleSheet("QToolButton { border: none; color: white; background-color: black; }")
        self.infoButton.setFixedSize(30, 30)
        self.infoButton.setToolTip(info_text)  # Tooltip text.
        self.header.addWidget(self.infoButton)  # Adding the information button to the header layout.

        # Adding a stretch factor to the header layout to ensure buttons are left-aligned.
        self.header.addStretch(1)

        self.layout.addLayout(self.header)  # Adding the header layout to the main vertical layout.

        # Content setup with a scroll area to allow scrolling through the content.
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setVisible(False)  # Initially hidden until toggled.
        self.scrollArea.setLayoutDirection(Qt.RightToLeft)  # Set layout direction to RightToLeft.
        self.contentWidget = QWidget()  # Widget that holds the content.
        self.contentWidget.setObjectName("viewport")  # Set object name for styling.
        self.contentWidget.setLayout(QVBoxLayout())  # Layout for the content.
        self.scrollArea.setWidget(self.contentWidget)  # Add content widget to the scroll area.

        # Custom styling for the scroll area and its components, particularly the vertical scrollbar.
        self.scrollArea.setStyleSheet("""
            QScrollArea {
                border: none;
                background: black;
            }
            QScrollArea QScrollBar:vertical {
                border: none;
                background: black;
                width: 10px;
                margin: 0px 0 0px 0;
                border-radius: 0px;
            }
            QScrollArea QScrollBar::handle:vertical {
                background-color: #5b5b5b;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollArea QScrollBar::handle:vertical:hover {
                background-color: #5b5b5b;
            }
            QScrollArea QScrollBar::add-line:vertical, QScrollArea QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
            QScrollArea QScrollBar::add-page:vertical, QScrollArea QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollArea QWidget#viewport {
                background: black;
            }
        """)  # Apply styling to scroll area and scrollbars.

        self.layout.addWidget(self.scrollArea)  # Add scroll area to the main layout.


    def onToggle(self):
        """
        Handles the toggle action for the collapsible section. It changes the visibility of the scroll area,
        updates the toggle button text, and optionally adjusts the layout based on the visibility status.

        - It checks the current state of the toggle button (checked or not) to determine if the section should be visible.
        - It sets the text of the toggle button to "-" if the section is visible, otherwise to "+".
        - If the section is made visible, it adjusts the scroll area's size and policy based on content height.
        - Finally, it updates the geometry of the widget and its parent layout, and adjusts the size of the top-level window if necessary.
        """
        isVisible = self.toggleButton.isChecked()  # Check if the toggle button is currently checked (True if visible).
        self.scrollArea.setVisible(isVisible)  # Set the visibility of the scroll area based on the toggle button's state.
        self.toggleButton.setText("-" if isVisible else "+")  # Set toggle button text based on visibility.

        if isVisible:
            self.adjustScrollArea()  # Adjust the scroll area dimensions and scrollbar visibility.
        self.updateGeometry()  # Update the widget's geometry to accommodate changes.

        if self.parentWidget():
            self.parentWidget().layout().activate()  # Refresh the layout of the parent widget if available.

        topLevelWindow = self.window()  # Get the top-level window that contains this widget.
        if topLevelWindow:
            topLevelWindow.adjustSize()  # Adjust the size of the top-level window to fit its new contents.

    def adjustScrollArea(self):
        """
        Adjusts the size and scrollbar visibility of the scroll area based on the content height relative to a predefined
        height threshold.

        - If the content height exceeds the threshold, a fixed size is set to the scroll area, and vertical scrollbars
        are shown as needed.
        - If the content height is below the threshold, the scroll area is resized to fit the content and vertical
        scrollbars are hidden.
        """
        contentHeight = self.contentWidget.sizeHint().height()  # Get the suggested height of the content widget.
        if contentHeight > self.height_threshold:
            self.scrollArea.setFixedSize(QSize(self.width(), self.height_threshold))  # Set fixed size to threshold height.
            self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show scrollbar if needed.
        else:
            self.scrollArea.setFixedSize(QSize(self.width(), contentHeight))  # Resize scroll area to content height.
            self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Hide vertical scrollbar.

    def contentLayout(self):
        """
        Provides access to the layout of the content widget inside the scroll area.

        Returns:
            QLayout: The layout manager of the content widget.
        """
        return self.contentWidget.layout()  # Return the layout used by the content widget.
    

class SettingsSection(CollapsibleSection):
    def __init__(self, title, parent=None):
        """
        Initializes a SettingsSection with a very large height threshold to prevent the scroll bar from appearing.
        Calls the parent constructor to set up basic collapsible functionality with an unusually high threshold.

        Args:
            title (str): The title of the collapsible section.
            parent (QWidget): The parent widget of this section, defaults to None.
        """
        # Initialize parent class with a large height threshold to avoid using a scroll bar in settings.
        super().__init__(title, parent, height_threshold=10000)
        # Initialize the user interface for the settings section.
        self.initUI()

    def initUI(self):
        """
        Sets up the user interface elements within the settings section, including a customized header and content area.

        - Clears existing widgets from the header and reconfigures it with settings-specific controls.
        - Sets up the scroll area and the content widget for holding setting inputs.
        - Populates the content area with example setting controls.
        """
        # Clear any existing widgets from the header.
        while self.header.count():
            item = self.header.takeAt(0)  # Remove the first item from the layout.
            if item.widget():
                item.widget().deleteLater()  # Properly delete the widget to free up resources.

        # Setup for the titleLabel specifically for the settings.
        self.titleLabel = QLabel("Settings")
        self.titleLabel.setStyleSheet("font-weight: bold; color: white;")

        # Adding and setting up the toggle button within the header.
        self.toggleButton = QPushButton("+")
        self.toggleButton.setFixedSize(30, 30)
        self.toggleButton.setStyleSheet("font-size: 18px; font-weight: bold; color: white; background-color: black;")
        self.toggleButton.setCheckable(True)
        self.toggleButton.setChecked(False)
        self.toggleButton.clicked.connect(self.onToggle)

        # Adding widgets to the header; adding stretch first pushes other elements to the right (reverse order).
        self.header.addStretch(1)
        self.header.addWidget(self.toggleButton)
        self.header.addWidget(self.titleLabel)

        # Initialize the scroll area and content widget for holding various settings controls.
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setVisible(False)
        self.contentWidget = QWidget()
        self.contentWidget.setLayout(QVBoxLayout())
        self.scrollArea.setWidget(self.contentWidget)

        # Setup example content widgets for various settings.
        # Voxel sizes input field
        self.voxelSizesEdit = QLineEdit("20.0, 25.0, 30.0, 35.0")
        self.contentLayout().addWidget(QLabel("Voxel Sizes:"))
        self.contentLayout().addWidget(self.voxelSizesEdit)
        applyLineEditStyle(self.voxelSizesEdit)  # Apply a custom style function to the QLineEdit widget.

        # RANSAC fitness threshold input field
        self.fitnessRansacEdit = QLineEdit("0.85")
        self.contentLayout().addWidget(QLabel("Desired Fitness RANSAC:"))
        self.contentLayout().addWidget(self.fitnessRansacEdit)
        applyLineEditStyle(self.fitnessRansacEdit)  # Similarly, apply styling.

        # ICP fitness thresholds input field
        self.fitnessIcpEdit = QLineEdit("0.65, 0.75, 0.85")
        self.contentLayout().addWidget(QLabel("Desired Fitness ICP:"))
        self.contentLayout().addWidget(self.fitnessIcpEdit)
        applyLineEditStyle(self.fitnessIcpEdit)  # Apply consistent styling across all QLineEdit widgets.

    def onToggle(self):
        """
        Toggles the visibility of the scroll area and all child widgets based on the state of the toggle button.
        Also updates the text of the toggle button to indicate the current state (collapsed or expanded).

        - Checks the toggle button's state and sets the visibility of the scroll area accordingly.
        - Iterates through all child widgets in the content layout, setting their visibility to match the scroll area.
        """
        isVisible = self.toggleButton.isChecked()  # Check if the toggle button is in the 'checked' state.
        self.scrollArea.setVisible(isVisible)  # Set the visibility of the scroll area based on the toggle button.
        self.toggleButton.setText("-" if isVisible else "+")  # Update the toggle button text to show current state.

        # Manually set the visibility of each widget in the content layout.
        for i in range(self.contentWidget.layout().count()):
            widget = self.contentWidget.layout().itemAt(i).widget()  # Access each widget in the layout.
            if widget:
                widget.setVisible(isVisible)  # Set the visibility of the widget.

    def adjustScrollArea(self):
        """
        Adjusts the size and scrollbar visibility of the scroll area based on the current content height,
        comparing it against a predefined height threshold.

        - If the content height exceeds the threshold, the scroll area is sized to the threshold and scrollbars are enabled.
        - If the content height is below the threshold, the scroll area is resized to fit the content and scrollbars are disabled.
        """
        contentHeight = self.contentWidget.sizeHint().height()  # Determine the height of the content widget.
        if contentHeight > self.height_threshold:
            self.scrollArea.setFixedSize(QSize(self.width(), self.height_threshold))  # Fix the size to the threshold height.
            self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Enable scrollbars only as needed.
        else:
            self.scrollArea.setFixedSize(QSize(self.width(), contentHeight))  # Resize scroll area to fit content height.
            self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Disable scrollbars.

    def getSettings(self):
        """
        Retrieves settings values from the user input fields, parses them, and returns them in appropriate data types.

        Returns:
            tuple: Contains three elements:
                - A list of voxel sizes (as floats),
                - The desired fitness for RANSAC (as a float),
                - A list of desired fitness values for ICP (as floats).
        """
        # Parse the text input for voxel sizes into a list of floats.
        voxel_sizes = list(map(float, self.voxelSizesEdit.text().split(',')))
        # Convert the text input for RANSAC fitness into a float.
        desired_fitness_ransac = float(self.fitnessRansacEdit.text())
        # Parse the text input for ICP fitness into a list of floats.
        desired_fitness_icp = list(map(float, self.fitnessIcpEdit.text().split(',')))

        return voxel_sizes, desired_fitness_ransac, desired_fitness_icp
    

def applyButtonStyle(button):
    """
    Applies a custom style to a QPushButton. The style includes visual aspects such as background color, text color,
    font weight, and others to enhance the appearance of buttons.

    Args:
        button (QPushButton): The button widget to apply the style to.

    - Sets the background color to red to make the button stand out.
    - Uses white text color for contrast against the red background.
    - Adds bold font weight for emphasis.
    - Rounds the corners with a radius of 12px.
    - Adds padding and margin to ensure the button's content is well-spaced and visually appealing.
    """
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
    """
    Sets a custom style for QLineEdit widgets to ensure visual consistency and readability.

    Args:
        lineEdit (QLineEdit): The line edit widget to which the style is applied.

    - Uses a light grey background for the normal state to make it visually distinct from other UI elements.
    - Specifies a darker text color for better readability.
    - Applies rounded corners to match stylistically with other rounded UI elements like buttons.
    - Adjusts padding and margins to align with the overall design scheme.
    - Modifies the appearance slightly for read-only mode to visually indicate that the widget is not editable.
    """
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

class LeftAlignDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        option.displayAlignment = Qt.AlignRight | Qt.AlignVCenter
        super(LeftAlignDelegate, self).paint(painter, option, index)

def applyComboBoxStyle(comboBox):
    """
    Sets a custom style for QComboBox widgets to ensure visual consistency and readability.

    Args:
        comboBox (QComboBox): The combo box widget to which the style is applied.
    """
    comboBox.setStyleSheet("""
        QComboBox {
            background-color: #000000;  /* Black background */
            color: #333;  /* Dark text color */
            border-radius: 12px;  /* Rounded corners */
            padding: 5px 30px 5px 15px;  /* Padding */
            font-size: 14px;  /* Font size */
            margin: 5px;  /* Margin */
        }
        QComboBox QAbstractItemView {
            background-color: #000000;  /* Black background for drop-down list */
            color: #ffffff;  /* White text color */
            border-radius: 6px;  /* Rounded corners for drop-down list */
            selection-background-color: #555555;  /* Background color for selected item */
            selection-color: #ffffff;  /* Text color for selected item */
            margin: 0px;
            padding: 0px;
        }
        QComboBox QAbstractItemView::item {
            background-color: #000000;  /* Black background for each item */
            color: #ffffff;  /* White text color */
        }
        QComboBox::drop-down {
            border: none;  /* No border for the drop-down button */
        }
        QComboBox::down-arrow {
            image: url("./ui/down_arrow.png");  /* Custom arrow icon */
            width: 14px;  /* Width of the arrow */
            height: 14px;  /* Height of the arrow */
            subcontrol-origin: padding;
            subcontrol-position: right center;
            right: 10px;  /* Move the arrow to the right */
        }
        QComboBox QAbstractItemView QScrollBar:vertical {
            border: none;
            background: #000000;  /* Black background for the scroll bar */
            width: 10px;
            margin: 0px 0 0px 0;
            border-radius: 0px;
        }
        QComboBox QAbstractItemView QScrollBar::handle:vertical {
            background-color: #5b5b5b;
            min-height: 30px;
            border-radius: 5px;
        }
        QComboBox QAbstractItemView QScrollBar::handle:vertical:hover {
            background-color: #5b5b5b;
        }
        QComboBox QAbstractItemView QScrollBar::add-line:vertical, QComboBox QAbstractItemView QScrollBar::sub-line:vertical {
            border: none;
            background: none;
            height: 0px;
        }
        QComboBox QAbstractItemView QScrollBar::add-page:vertical, QComboBox QAbstractItemView QScrollBar::sub-page:vertical {
            background: none;
        }
        QComboBox::item:selected {
            background-color: #555555;  /* Selected item background color */
            color: #ffffff;  /* Selected item text color */
        }
        QComboBox QAbstractItemView::item:selected {
            background-color: #555555;  /* Selected item background color */
            color: #ffffff;  /* Selected item text color */
        }
        QComboBox:editable {
            background-color: #ffffff;  /* White background for the line edit part */
            color: #000000;  /* Black text color */
        }
        QComboBox QLineEdit {
            background-color: #ffffff;  /* White background for the line edit part */
            color: #000000;  /* Black text color */
        }
    """)
    comboBox.setItemDelegate(LeftAlignDelegate(comboBox))  # Set item delegate for alignment
    comboBox.setEditable(True)  # Make the combo box editable to style the line edit part
    comboBox.lineEdit().setReadOnly(True)  # Make the line edit read-only to prevent user editing

def applyTextAndScrollBarStyle(widget):
    """
    Configures custom styles for QTextEdit widgets including the main text area and its scrollbars. The style 
    ensures consistency in visual design across text-editing components.

    Args:
        widget (QTextEdit): The text edit widget to which the styles will be applied.

    - Sets the primary text color, padding, margin, and background color for readability and aesthetic alignment with the app's design.
    - Customizes the scrollbar appearance to blend seamlessly with the QTextEdit, including modifications for hover states.
    - Ensures that read-only mode and other interactive elements like scrollbars have distinctive styles to indicate their state and function.
    """
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

def setupInputField(layout, label_text, placeholder_text):
    """
    Helper method to set up a label and QLineEdit for user input.

    Args:
        layout (QVBoxLayout): The layout into which the elements are to be integrated.
        label_text (str): The text to display on the label.
        placeholder_text (str): The placeholder text for the QLineEdit.
    """
    # Create a label with the provided text and add it to the layout
    label = QLabel(label_text)
    label.setAlignment(Qt.AlignLeft)
    layout.addWidget(label)

    # Create a QLineEdit with the provided placeholder text and add it to the layout
    line_edit = QLineEdit()
    line_edit.setPlaceholderText(placeholder_text)
    applyLineEditStyle(line_edit)
    layout.addWidget(line_edit)

    return line_edit  # Return the QLineEdit for further use

def createLoadingWheel(parent_section):
    """
    Creates a loading wheel animation within a given section of the UI. This is typically used to indicate
    that a process is ongoing.

    Args:
        parent_section: The parent widget or section where the loading animation will be displayed.

    Returns:
        tuple: Contains the QLabel configured with the loading animation and the QMovie object for the animation.

    - Initializes a QHBoxLayout to center the loading label horizontally.
    - Sets up a QLabel and assigns a QMovie with a gif animation for visual representation of loading.
    - The label is initially hidden until the animation needs to be shown.
    """
    wheelLayout = QHBoxLayout()  # Create a horizontal layout to hold the loading animation.
    wheelLayout.addStretch()  # Add a stretch to center the label.

    loadingLabel = QLabel()  # Create a label to hold the animation.
    loadingMovie = QMovie("./ui/spinning_wheel_v2_60.gif")  # Load the animation gif.
    loadingLabel.setMovie(loadingMovie)  # Assign the movie to the label.
    loadingLabel.setAlignment(Qt.AlignCenter)  # Center the label within its available space.
    loadingLabel.setFixedSize(25, 25)  # Set a fixed size for the label.
    loadingMovie.setScaledSize(loadingLabel.size())  # Ensure the movie scales to fit the label.
    loadingLabel.hide()  # Initially hide the label; it will be shown during loading.

    wheelLayout.addWidget(loadingLabel)  # Add the label to the layout.
    wheelLayout.addStretch()  # Add another stretch to ensure the label stays centered.
    parent_section.contentLayout().addLayout(wheelLayout)  # Add the entire layout to the parent section.

    return loadingLabel, loadingMovie

def startLoadingAnimation(label, movie, scrollArea):
    """
    Starts the loading animation by showing the label, starting the movie, and adjusting the scroll area
    to accommodate the animation.

    Args:
        label (QLabel): The label containing the loading animation.
        movie (QMovie): The animation to be played.
        scrollArea (QScrollArea): The scroll area to be adjusted in height.

    - The label is shown and the animation is started.
    - The height of the scroll area is increased to provide space for the animation.
    """
    label.show()  # Make the label visible.
    movie.start()  # Start playing the loading animation.
    current_size = scrollArea.size()  # Get the current size of the scroll area.
    updated_height = current_size.height() + 20  # Increase the height to make room for the animation.
    scrollArea.setFixedSize(current_size.width(), updated_height)  # Apply the new size to the scroll area.

def stopLoadingAnimation(label, movie, scrollArea):
    """
    Stops the loading animation by stopping the movie, hiding the label, and restoring the original
    height of the scroll area.

    Args:
        label (QLabel): The label containing the loading animation.
        movie (QMovie): The animation to be stopped.
        scrollArea (QScrollArea): The scroll area to be adjusted in height.

    - The animation is stopped and the label is hidden.
    - The height of the scroll area is decreased to its original size.
    """
    movie.stop()  # Stop the animation.
    label.hide()  # Hide the label.
    current_size = scrollArea.size()  # Get the current size of the scroll area.
    updated_height = current_size.height() - 20  # Decrease the height to remove the space for the animation.
    scrollArea.setFixedSize(current_size.width(), updated_height)  # Apply the new size to the scroll area.







