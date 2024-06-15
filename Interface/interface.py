import sys
import os
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel, QProgressBar, QVBoxLayout, QFileDialog, QMainWindow, QMessageBox, QTableWidget, QTableWidgetItem
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

        # Reference your table widgets here, after loading the UI
        self.tables = {
            1: self.findChild(QTableWidget, "tableMatrix05_6"),  # Matrix (01x01)
            2: self.findChild(QTableWidget, "tableMatrix05_2"),  # Matrix (02x02)
            4: self.findChild(QTableWidget, "tableMatrix05_3"),  # Matrix (04x04)
            8: self.findChild(QTableWidget, "tableMatrix05_5"),  # Matrix (08x08)
            16: self.findChild(QTableWidget, "tableMatrix05_7"), # Matrix (16x16)
            32: self.findChild(QTableWidget, "tableMatrix05_8")  # Matrix (32x32)
        }

        self.show()

    def initProcessing(self, dataset_path):
        files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
        for file in files:
            print(dataset_path +"/"+ file)
        
        # Load and display the image
        image_path = "5.png"  # Change this to the path of your image
        self.load_image(image_path)

    def populateInterface(self, original_image_path):
        imagemCinza = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        histogramaCinza = interface.grayHistogram(imagemCinza)
        histogramaColorido2D = interface.colorHistogram(original_image_path)

    def load_image(self, image_path):
        imagemOriginal = cv2.imread(image_path)
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
        imagemOriginal = cv2.imread(image_path)
        hu_moments = self.showHu_Moments(imagemOriginal)
        populate_channel_table(self, hu_moments)
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

    def plotMatrix(self, imagemCinza):
        # Distances for the co-occurrence matrices
        distances = [1, 2, 4, 8, 16, 32]

        # Calculating matrices
        co_occurrence_matrices = {}
        for distance in distances:
            co_occurrence_matrices[distance] = interface.compute_co_occurrence_matrix(imagemCinza, distance)
            self.displayCoOccurrenceMatrix(distance, co_occurrence_matrices[distance])
            
        haralick_features = {}
        for distance, co_occurrence_matrix in co_occurrence_matrices.items():
            features = {
                'Homogeneity': interface.calculate_homogeneity(co_occurrence_matrix),
                'Contrast': interface.calculate_contrast(co_occurrence_matrix),
                'Entropy': interface.calculate_entropy(co_occurrence_matrix)
            }
            haralick_features[distance] = features
        
        self.populate_tables(haralick_features)

    def showHu_Moments(self, imagemOriginal):
        hu_moments = interface.extract_hu_moments(imagemOriginal)
        return hu_moments


    def populate_tables(self, haralick_features):
        for distance, features in haralick_features.items():
            table = self.tables[distance]
            if table:
                table.setRowCount(1)
                table.setColumnCount(3)
                table.setHorizontalHeaderLabels(["Contrast", "Entropy", "Homogeneity"])
                print(f"Distance {distance}: Contrast = {features['Contrast']}, Entropy = {features['Entropy']}, Homogeneity = {features['Homogeneity']}")

                table.setItem(0, 0, QTableWidgetItem(str(features['Contrast'])))
                table.setItem(0, 1, QTableWidgetItem(str(features['Entropy'])))
                table.setItem(0, 2, QTableWidgetItem(str(features['Homogeneity'])))

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

    def triggerImportButton(self):
        dataset_path = self.getDatasetFolderName()
        if dataset_path:
            self.initProcessing(dataset_path)
        else:
            print("Processamento cancelado...")

    def getDatasetFolderName(self):
        folderName = QFileDialog.getExistingDirectory(self, caption="Select a folder")
        if folderName:
            if self.open_progress_dialog():
                self.frameContentAfterImport.show()
            else:
                folderName = None
        return folderName

    def open_progress_dialog(self):
        dialog = ProgressDialog(self)
        dialog.move(self.rect().center())
        dialog.exec()
        return dialog.dialog_accepted
    

def populate_channel_table(self, hu_moments):
    table = self.findChild(QTableWidget, "tableChannel")
    if table:
        table.setRowCount(7)  # Set 7 rows for 7 Hu moments
        table.setColumnCount(4)  # Set 4 columns for 4 channels

        table.setHorizontalHeaderLabels(["Channel G", "Channel H", "Channel S", "Channel V"])

        # Populate the table with values
        for i in range(7):  # Iterate over each Hu moment
            # Each Hu moment has values for G, H, S, V
            g_value = hu_moments[i]
            h_value = hu_moments[i + 7]
            s_value = hu_moments[i + 14]
            v_value = hu_moments[i + 21]

            # Insert values into the table
            table.setItem(i, 0, QTableWidgetItem(f"{g_value:.4f}"))
            table.setItem(i, 1, QTableWidgetItem(f"{h_value:.4f}"))
            table.setItem(i, 2, QTableWidgetItem(f"{s_value:.4f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{v_value:.4f}"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    UIWindow = UI()
    app.exec()
