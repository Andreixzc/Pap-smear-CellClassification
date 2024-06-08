import cv2
from InterfaceUtils import InterfaceUtils

# Carregar a imagem usando OpenCV
image = cv2.imread("5.png", cv2.IMREAD_GRAYSCALE)

# Criar uma instância da classe InterfaceUtils
interface = InterfaceUtils()
print("Momentos de Hu da imagem:")
print(interface.calculate_hu_moments(image))

# Definir os valores de i desejados
is_values = [1, 2, 4, 8, 16, 32]

# Armazenar as matrizes de co-ocorrência para cada valor de i
co_occurrence_matrices = []

# Calcular as matrizes de co-ocorrência para cada valor de i
for i in is_values:
    co_matrices = interface.co_occurrence_matrices(image, levels=16, distances=[i], angles=[0])
    co_occurrence_matrices.append(co_matrices)

# Iterar sobre as matrizes de co-ocorrência e calcular os descritores de Haralick para cada uma
for i, co_matrices in zip(is_values, co_occurrence_matrices):
    print(f"Para i = {i}:")
    for key, matrix in co_matrices.items():
        distance, angle = key
        print(f"\tDescritores de Haralick para a matriz de co-ocorrência para distância {distance} e ângulo {angle}:")
        entropy, homogeneity, contrast = interface.haralick_descriptors(matrix)
        print(f"\tEntropia: {entropy}")
        print(f"\tHomogeneidade: {homogeneity}")
        print(f"\tContraste: {contrast}")
        print()
        