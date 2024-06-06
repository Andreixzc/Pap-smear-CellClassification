# main.py

import cv2
from hu_moments_extractor import HuMomentsExtractor

# Exemplo de uso da classe para extrair momentos Hu de uma única imagem
image_path = '10.png'
image = cv2.imread(image_path)
if image is not None:
    extractor = HuMomentsExtractor()
    hu_moments = extractor.extract_hu_moments(image)
    print(type(hu_moments))
    print("Momentos Hu extraídos:", hu_moments)
else:
    print("Erro ao carregar a imagem.")

def hello():
    print("Hello, World!")