import tkinter as tk
from tkinter import filedialog

class UserInterface:
    def __init__(self):
        self.selected_file_path = None

    def select_file(self):
        # Ensure the tkinter root window is initialized in the foreground
        root = self.initialize_tkinter_root()
        
        # Use filedialog to select the .ply file from the flowvisdata folder
        file_path = filedialog.askopenfilename(initialdir="D:/flowvis_data",
                                               title="Select a .ply file",
                                               filetypes=[("PLY files", "*.ply")])

        # Destroy the hidden Tkinter root window
        root.destroy()

        # Check if a file was selected
        if file_path:
            self.selected_file_path = file_path
        else:
            print("No file selected.")

    @staticmethod
    def initialize_tkinter_root():
        # Create a hidden Tkinter root window
        root = tk.Tk()
        root.withdraw()
        return root

if __name__ == "__main__":
    ui = UserInterface()
    ui.select_file()

    if ui.selected_file_path:
        print("Selected file path:", ui.selected_file_path)
    else:
        print("No file selected.")
