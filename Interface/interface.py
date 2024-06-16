import sys
import os
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel, QProgressBar, QVBoxLayout, QFileDialog, QMainWindow, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QGraphicsScene
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6 import uic
import cv2
import numpy as np
from InterfaceUtils import InterfaceUtils
from matplotlib import pyplot as plt

interface = InterfaceUtils()
currentImg = ""
dataset = []

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

        self.zoomInButton = self.findChild(QPushButton, 'zoomInButton')
        self.zoomOutButton = self.findChild(QPushButton, 'zoomOutButton')
        self.originalImageLabel = self.findChild(QLabel, 'originalImageLabel')
        self.grayscaleImageLabel = self.findChild(QLabel, 'greyscaleImageLabel')

        self.nextImageButton = self.findChild(QPushButton, 'nextImageButton')
        self.previousImageButton = self.findChild(QPushButton, 'previousImageButton')

        # Connect buttons to methods
        self.zoomInButton.clicked.connect(self.zoom_in)
        self.zoomOutButton.clicked.connect(self.zoom_out)
        self.nextImageButton.clicked.connect(self.nextImage)
        self.previousImageButton.clicked.connect(self.previousImage)

        # Connect buttons to methods
        self.zoomInButton.clicked.connect(self.zoom_in)
        self.zoomOutButton.clicked.connect(self.zoom_out)

        self.originalScene = QGraphicsScene()
        self.grayscaleScene = QGraphicsScene()

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
        self.tables["tableChannel_2"] = self.findChild(QTableWidget, "tableChannel_2")
        
        self.predictButton = self.findChild(QPushButton, "predictButton")
        self.predictButton.clicked.connect(self.predictClass)
        self.show()



    def nextImage(self):
        global currentImg
        global dataset
        currentImg = (currentImg+1) % len(dataset)
        print(f"Next Image: {currentImg}")
        self.load_image(dataset[currentImg])

    def previousImage(self):
        global currentImg
        global dataset
        currentImg = (currentImg-1) % len(dataset)
        print(f"Prev Image: {currentImg}")
        self.load_image(dataset[currentImg])

    def predictClass(self):
        global currentImg
        global dataset
        img = cv2.imread(dataset[currentImg])
        result = interface.predict(img, dataset[currentImg])
        previsao_binario = result["previsao_binario_modelo_xgboost"]
        previsao_multiclasse = result["previsao_multiclasse_modelo_xgboost"]
        prediction_binary = result["previsao_binario_modelo_effnet"]
        prediction_multi = result["previsao_multiclasse_modelo_effnet"]

        # Update tableChannel_2 with the predictions
        table = self.tables["tableChannel_2"]
        if table:
            table.setItem(0, 0, QTableWidgetItem(str(prediction_binary)))  # EFF NET 2
            table.setItem(1, 0, QTableWidgetItem(str(prediction_multi)))   # EFF NET 6
            table.setItem(2, 0, QTableWidgetItem(str(previsao_binario)))   # XGBOOST
            table.setItem(3, 0, QTableWidgetItem(str(previsao_multiclasse)))  # XGBOOST 6


    
    def initProcessing(self, dataset_path):
        global currentImg
        global dataset

        png_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.png'):
                    png_files.append(os.path.join(root, file))
        # files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
        # for file in files:
        #     print(dataset_path +"/"+ file)
        
        # Load and display the image
        # image_path = "5.png"  # Change this to the path of your image
        dataset = png_files
        currentImg = 0
        self.load_image(dataset[currentImg])

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
            self.originalScene.addPixmap(pixmap)
            self.originalImageView.setScene(self.originalScene)
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
                self.grayscaleScene.addPixmap(grayPixmap)
                self.grayscaleImageView.setScene(self.grayscaleScene)
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
                table.setRowCount(3)
                table.setColumnCount(1)
                table.horizontalHeader().setFixedHeight(0)
                print(f"Distance {distance}: Contrast = {features['Contrast']}, Entropy = {features['Entropy']}, Homogeneity = {features['Homogeneity']}")

                table.setItem(0, 0, QTableWidgetItem(str(features['Contrast'])))
                table.setItem(1, 0, QTableWidgetItem(str(features['Entropy'])))
                table.setItem(2, 0, QTableWidgetItem(str(features['Homogeneity'])))

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
        plt.imshow(hist, interpolation='nearest', aspect='auto')
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


    def zoom_out(self):
        self.grayscaleImageView.scale(0.95, 0.95)
        self.originalImageView.scale(0.95, 0.95)

    def zoom_in(self):
        self.grayscaleImageView.scale(1.05, 1.05)
        self.originalImageView.scale(1.05, 1.05)
        
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
