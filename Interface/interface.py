import sys
import os
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel, QProgressBar, QVBoxLayout, QFileDialog, QMainWindow, QMessageBox
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer
from PyQt6 import uic
import cv2
import numpy as np
from InterfaceUtils import InterfaceUtils
from matplotlib import pyplot as plt
interface = InterfaceUtils()


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
            print(dataset_path +"/"+ file)
        
        # Load and display the image
        image_path = "5.png"  # Change this to the path of your image
        self.load_image(image_path)

    
    def populateInterface (self, original_image_path):
        imagemCinza = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        histogramaCinza = interface.grayHistogram(imagemCinza)
        histogramaColorido2D = interface.colorHistogram(original_image_path)
    
    def load_image(self, image_path):
        self.groupBoxImages.setTitle(image_path)
        
        
        # Carregar imagem principal
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.originalImageLabel.setPixmap(pixmap)
            self.originalImageLabel.setScaledContents(True)
            self.originalImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            print("Failed to load image:", image_path)
        self.loadHistogram(image_path)
        
        
        # Carregar imagem em cinza
        imagemCinza = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.plotMatrix(imagemCinza)
        self.loadGrayHistogram(imagemCinza)
        if imagemCinza is None:
            print("Failed to load image:", image_path)
        else:
            # Convert the OpenCV image (numpy array) to QImage
            height, width = imagemCinza.shape
            bytesPerLine = width
            q_image = QImage(imagemCinza.data, width, height, bytesPerLine, QImage.Format.Format_Grayscale8)

            # Convert QImage to QPixmap
            grayPixmap = QPixmap.fromImage(q_image)

            # Check if pixmap is not null and set it to the label
            if not grayPixmap.isNull():
                self.greyscaleImageLabel.setPixmap(grayPixmap)
                self.greyscaleImageLabel.setScaledContents(True)
                self.greyscaleImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                print("Failed to load image:", image_path)
        #self.loadGrayHistogram(imagemCinza)

    def plotMatrix(self, imagemCinza):
        # Distancias da matriz
        distances = [1, 2, 4, 8, 16, 32]

        # Calculando matriz
        co_occurrence_matrices = {}
        for distance in distances:
            co_occurrence_matrices[distance] = interface.compute_co_occurrence_matrix(imagemCinza, distance)
            self.displayCoOccurrenceMatrix(distance = distance, matrix = co_occurrence_matrices[distance])


    def displayCoOccurrenceMatrix(self, distance, matrix):
        # Normalize the matrix values to the range [0, 255]
        norm_matrix = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the matrix to 8-bit unsigned integers
        matrix_image = np.uint8(norm_matrix)

        # Resize the matrix image for better visibility (optional)
        matrix_image = cv2.resize(matrix_image, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Convert the matrix image to QImage
        height, width = matrix_image.shape
        bytesPerLine = width
        q_image = QImage(matrix_image.data, width, height, bytesPerLine, QImage.Format.Format_Grayscale8)

        # Convert QImage to QPixmap and display it
        pixmap = QPixmap.fromImage(q_image)
        if (distance == 1):
            self.matrix01ImageLabel.setPixmap(pixmap)
            self.matrix01ImageLabel.setScaledContents(True)
            self.matrix01ImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        elif (distance == 2):
            self.matrix02ImageLabel.setPixmap(pixmap)
            self.matrix02ImageLabel.setScaledContents(True)
            self.matrix02ImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        elif (distance == 4):
            self.matrix04ImageLabel.setPixmap(pixmap)
            self.matrix04ImageLabel.setScaledContents(True)
            self.matrix04ImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        elif (distance == 8):
            self.matrix08ImageLabel.setPixmap(pixmap)
            self.matrix08ImageLabel.setScaledContents(True)
            self.matrix08ImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        elif (distance == 16):
            self.matrix16ImageLabel.setPixmap(pixmap)
            self.matrix16ImageLabel.setScaledContents(True)
            self.matrix16ImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        elif (distance == 32):
            self.matrix32ImageLabel.setPixmap(pixmap)
            self.matrix32ImageLabel.setScaledContents(True)
            self.matrix32ImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            

    def loadGrayHistogram(self, imagemCinza):
        hist = interface.grayHistogram(imagemCinza)
        plt.figure()
        plt.plot(hist, color='black')
        plt.title("Histograma em Tons de Cinza")
        plt.xlabel("Intensidade")
        plt.ylabel("Número de Pixels")
        
        # Save the plot as an image
        plt.savefig('ghistogram.png')
        plt.close()

        # Load the histogram image
        hist_image = cv2.imread('ghistogram.png')
        if hist_image is None:
            print("Failed to load histogram image")
        else:
            # Convert the histogram image to QImage and display it
            height, width, channel = hist_image.shape
            bytesPerLine = 3 * width
            q_image = QImage(hist_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.grayscaleHistogramLabel.setPixmap(pixmap)
            self.grayscaleHistogramLabel.setScaledContents(True)
            self.grayscaleHistogramLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #histogramaColorido2D = interface.colorHistogram(original_image_path)

    def loadHistogram(self, image_path):
        image = cv2.imread(image_path)
        hist = interface.colorHistogram(image)
        plt.figure()
        plt.plot(hist)
        plt.title('Histograma 2D para H e V')
        plt.xlabel('Valores de V')
        plt.ylabel('Valores de H')
        
        # Save the plot as an image
        plt.savefig('histogram.png')
        plt.close()

        # Load the histogram image
        hist_image = cv2.imread('histogram.png')
        if hist_image is None:
            print("Failed to load histogram image")
        else:
            # Convert the histogram image to QImage and display it
            height, width, channel = hist_image.shape
            bytesPerLine = 3 * width
            q_image = QImage(hist_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.hsvHistogramImageLabel.setPixmap(pixmap)
            self.hsvHistogramImageLabel.setScaledContents(True)
            self.hsvHistogramImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

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


