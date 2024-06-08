import cv2
from InterfaceUtils import InterfaceUtils
interface = InterfaceUtils()


image = cv2.imread("5.png")

print("Momentos de Hu da imagem:")
print(interface.extract_hu_moments(image))

# Carregar a imagem usando OpenCV
image = cv2.imread("5.png", cv2.IMREAD_GRAYSCALE)


# Definir os valores de i desejados
is_values = [1, 2, 4, 8, 16, 32]

# Armazenar as matrizes de co-ocorrência para cada valor de i
co_occurrence_matrices = []

# Calcular as matrizes de co-ocorrência para cada valor de i
for i in is_values:
    co_matrix = interface.co_occurrence_matrices(image, levels=16, distances=[i], angles=[0])
    co_occurrence_matrices.append(co_matrix[(i, 0)])  # Acessa a matriz para o ângulo 0 para o valor de i atual

# Iterar sobre as matrizes de co-ocorrência e calcular os descritores de Haralick para cada uma
for i, co_matrix in zip(is_values, co_occurrence_matrices):
    print(f"Para i = {i}:")
    entropy, homogeneity, contrast = interface.haralick_descriptors(co_matrix)
    print(f"\tDescritores de Haralick para a matriz de co-ocorrência para i = {i}:")
    print(f"\tEntropia: {entropy}")
    print(f"\tHomogeneidade: {homogeneity}")
    print(f"\tContraste: {contrast}")
    print()