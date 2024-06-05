import cv2
import numpy as np

class HuMomentsExtractor:
    def __init__(self):
        pass

    def calculate_hu_moments(self, image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        moments = cv2.moments(gray_image)
        hu_moments = cv2.HuMoments(moments)
        hu_moments = -np.sign(hu_moments) * np.log(np.abs(hu_moments))

        return hu_moments.flatten()

    def extract_hu_moments(self, image):
        gray_hu_moments = self.calculate_hu_moments(image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        
        h_moments = self.calculate_hu_moments(h)
        s_moments = self.calculate_hu_moments(s)
        v_moments = self.calculate_hu_moments(v)

        hsv_hu_moments = np.concatenate((h_moments, s_moments, v_moments))
        all_moments = np.concatenate((gray_hu_moments, hsv_hu_moments))

        return all_moments