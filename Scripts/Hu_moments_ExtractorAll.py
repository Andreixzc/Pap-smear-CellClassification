import cv2
import numpy as np
import csv
import os

# Este código extrai todos os hu moments do dataset na pasta especificada, e salva em um arquivo csv.
# Temos que modificar esse código pra que ele extraia os hu moments de uma única imagem, pra mostrar na interface, e jogar nos
# classificadores XGB.

# Lembrar de para cada imagem, gerar dois 'vetores', um para partindo do pressuposto da imagem ter multi classes
# e o outro para a imagem ter apenas duas classes (binário).

# Labels: ASC-H, ASC-US, HSIL, LSIL, SCC, e intraepithelial lesion
# Labels binárias: 0 = intraepithelial lesion, 1 = demais.


def calculate_hu_moments(image):
    # Verificando se a imagem é colorida
    if len(image.shape) == 3:
        # Convertendo a imagem para escala de cinza
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Calculando os momentos invariantes de Hu
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments)
    # Fazendo a escala logarítmica dos momentos para torná-los invariantes à escala
    epsilon = 1e-10  # Um pequeno valor para substituir zeros
    hu_moments = -np.sign(hu_moments) * np.log(np.abs(hu_moments) + epsilon)

    return hu_moments.flatten()


def calculate_hsv_moments(image):
    # Convertendo a imagem para o espaço de cores HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Separando os canais de cor
    h, s, v = cv2.split(hsv_image)

    # Calculando os momentos invariantes de Hu para cada canal
    h_moments = calculate_hu_moments(h)
    s_moments = calculate_hu_moments(s)
    v_moments = calculate_hu_moments(v)

    return np.concatenate((h_moments, s_moments, v_moments))


instancesCount = 0
data = []
basePath = "28-05-2024/"
labels = [
    "ASC-H",
    "ASC-US",
    "HSIL",
    "LSIL",
    "Negative for intraepithelial lesion",
    "SCC",
]
num_moments = 28
csv_columns = [f"hu_moment_{i}" for i in range(1, num_moments + 1)] + ["label"]

for label in labels:
    label_path = os.path.join(basePath, label)
    for filename in os.listdir(label_path):
        image_path = os.path.join(label_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            gray_hu_moments = calculate_hu_moments(image)
            hsv_hu_moments = calculate_hsv_moments(image)
            grey_hsv_hu_moments = np.concatenate(
                (gray_hu_moments, hsv_hu_moments)
            )
            row_data = list(grey_hsv_hu_moments) + [label]
            data.append(row_data)
            instancesCount += 1

with open("hu_moments.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(csv_columns)
    writer.writerows(data)
