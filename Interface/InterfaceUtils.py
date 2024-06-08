import cv2
import numpy as np

class InterfaceUtils:
    @staticmethod
    def calculate_hu_moments(image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        moments = cv2.moments(gray_image)
        hu_moments = cv2.HuMoments(moments)
        hu_moments = -np.sign(hu_moments) * np.log(np.abs(hu_moments))

        return hu_moments.flatten()

    @staticmethod
    def extract_hu_moments(image):
        gray_hu_moments = InterfaceUtils.calculate_hu_moments(image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        
        h_moments = InterfaceUtils.calculate_hu_moments(h)
        s_moments = InterfaceUtils.calculate_hu_moments(s)
        v_moments = InterfaceUtils.calculate_hu_moments(v)

        hsv_hu_moments = np.concatenate((h_moments, s_moments, v_moments))
        all_moments = np.concatenate((gray_hu_moments, hsv_hu_moments))

        return all_moments



    @staticmethod
    def co_occurrence_matrices(image, levels=16, distances=[1], angles=[0]):
        # Quantize a imagem para o número de níveis de cinza especificado
        quantized_image = np.floor(image / (256 / levels)).astype(np.uint8)
        
        # Inicializar as matrizes de co-ocorrência para cada ângulo e distância
        co_occurrence_matrices = {}
        for distance in distances:
            for angle in angles:
                co_matrix = np.zeros((levels, levels), dtype=np.uint32)
                rows, cols = quantized_image.shape
                for i in range(rows):
                    for j in range(cols):
                        for d in range(1, distance + 1):
                            # Calcula as coordenadas do pixel vizinho
                            row_offset = int(d * np.sin(angle))
                            col_offset = int(d * np.cos(angle))
                            new_row = i + row_offset
                            new_col = j + col_offset
                            # Verifica se o pixel vizinho está dentro da imagem
                            if 0 <= new_row < rows and 0 <= new_col < cols:
                                # Incrementa a entrada correspondente na matriz de co-ocorrência
                                co_matrix[quantized_image[i, j], quantized_image[i, j]] += 1  # Apenas o valor [i][i] é atualizado
                # Normaliza a matriz de co-ocorrência para que a soma de todos os elementos seja 1
                co_matrix = co_matrix.astype(np.float64)
                co_matrix /= np.sum(co_matrix)
                co_occurrence_matrices[(distance, angle)] = co_matrix
        return co_occurrence_matrices

    @staticmethod
    def haralick_descriptors(co_occurrence_matrix):
        # Normaliza a matriz de co-ocorrência para que a soma de todos os elementos seja 1
        co_occurrence_matrix_normalized = co_occurrence_matrix / np.sum(co_occurrence_matrix)
        
        # Calcula as propriedades da matriz de co-ocorrência
        contrast = np.sum(np.multiply(np.square(np.arange(co_occurrence_matrix.shape[0]) - np.arange(co_occurrence_matrix.shape[1])), co_occurrence_matrix_normalized))
        entropy = -np.sum(co_occurrence_matrix_normalized * np.log2(co_occurrence_matrix_normalized + 1e-10))
        homogeneity = np.sum(co_occurrence_matrix_normalized / (1 + np.abs(np.arange(co_occurrence_matrix.shape[0]) - np.arange(co_occurrence_matrix.shape[1]))))

        return entropy, homogeneity, contrast

        



