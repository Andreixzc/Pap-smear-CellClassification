import sys
import os
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel, QProgressBar, QVBoxLayout, QFileDialog, QMainWindow, QMessageBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer
from PyQt6 import uic


class ProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Aviso")
        self.setGeometry(0, 0, 216, 106)

        self.dialog_accepted = False

        self.label = QLabel("Deseja iniciar o processamento ?", self)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)

        self.button_box = QDialogButtonBox(Qt.Orientation.Horizontal)
        self.button_box.setStandardButtons(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.show_progress_bar)  # Connect OK button to show progress bar
        self.button_box.rejected.connect(self.reject)  # Connect Cancel button to reject dialog

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.button_box)

        self.progress_bar.hide()  # Hide progress bar initially
    
    def status_window(self):
        return self.button_box.accepted()
        

    def show_progress_bar(self):
        self.dialog_accepted = True
        self.progress_bar.show()

        # Simulate progress update
        self.progress = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(50)  # Update progress every 50 milliseconds
        
    def update_progress(self):
        self.progress += 1
        self.progress_bar.setValue(self.progress)
        if self.progress >= 100:
            self.timer.stop()
            self.accept()  # Close dialog when progress reaches 100%

    def reject(self):
        super().reject()
        QMessageBox.critical(self, "Error", "Process canceled or encountered an error")

    def accept(self):
        super().accept()
        QMessageBox.information(self, "Success", "Dataset loaded successfully")

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        
        # Load UI File
        uic.loadUi("main.ui", self)
        

        # Define our widgets
        self.datasetPath = self.actionDataset.triggered.connect(self.triggerImportButton)

        self.frameContentAfterImport.hide()


        self.show()


    def initProcessing(self, dataset_path):
        files = [f for f in os.listdir(dataset_path) if os.path.isfile(f)]
        for file in files:
            print(file)
        
        # Load and display the image
        image_path = "teste.png"  # Change this to the path of your image
        self.load_image(image_path)
    
    def load_image(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.originalImageLabel.setPixmap(pixmap)
            self.originalImageLabel.setScaledContents(True)
            self.originalImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            print("Failed to load image:", image_path)

    def triggerImportButton (self):
        dataset_path = self.getDatasetFolderName()
        if (dataset_path != None):
            self.initProcessing(dataset_path)
        else:
            print("Processamento cancelado...")

    def getDatasetFolderName(self):
        folderName = QFileDialog.getExistingDirectory(self, caption="Select a folder")
        if folderName:
            if (self.open_progress_dialog()):
                self.frameContentAfterImport.show()
            else:
                folderName = None
        
            
        return folderName

    def open_progress_dialog(self):
        dialog = ProgressDialog(self)
        # Move the dialog to the adjusted position
        dialog.move(self.rect().center())
        dialog.exec()
        return dialog.dialog_accepted




if __name__ == "__main__":
    app = QApplication(sys.argv)
    UIWindow = UI()
    app.exec()


