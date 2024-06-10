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
    def compute_co_occurrence_matrix(img, distance):
        co_occurrence_matrix = np.zeros((16, 16), dtype=np.int32)

        limit = img.shape[1] - distance
        for y in range(img.shape[0]):
            for x in range(limit):
                g1 = img[y, x] // (256 // 16)
                g2 = img[y, x + distance] // (256 // 16)
                co_occurrence_matrix[g1, g2] += 1

        return co_occurrence_matrix

    @staticmethod
    def calculate_homogeneity(co_matrix):
        homogeneity = 0.0

        size = co_matrix.shape[0]
        for i in range(size):
            for j in range(size):
                homogeneity += co_matrix[i, j] / (1.0 + abs(i - j))

        return homogeneity
    

    @staticmethod
    def calculate_contrast(co_matrix):
        contrast = 0.0

        size = co_matrix.shape[0]
        for i in range(size):
            for j in range(size):
                contrast += (i - j) * (i - j) * co_matrix[i, j]

        return contrast
    
    @staticmethod
    def calculate_entropy(co_matrix):
        entropy = 0.0

        size = co_matrix.shape[0]
        for i in range(size):
            for j in range(size):
                value = co_matrix[i, j]
                if value > 0:
                    entropy -= value * np.log2(value)

        return entropy

        



