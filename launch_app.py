from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
# from main_baseline import MainApp       # import either baseline or development app
from main_development import MainApp

# Run the script as the main program.
if __name__ == "__main__":
    # Initialize the QApplication object
    app = QApplication([])  # The list can hold command line arguments passed to the application.

    # Set a dark theme for the entire application by modifying the style sheet of the application.
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
    window.show()  # Ensure the main window is visible

    # Execute the application's main loop
    app.exec()
