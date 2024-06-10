import cv2
from InterfaceUtils import InterfaceUtils
interface = InterfaceUtils()

image = cv2.imread("5.png")

print("Momentos de Hu da imagem:")
print(interface.extract_hu_moments(image))

# Carregar a imagem usando OpenCV
image = cv2.imread("5.png", cv2.IMREAD_GRAYSCALE)

# Specify distances
distances = [1, 2, 4, 8, 16, 32]

# Calculate co-occurrence matrices
co_occurrence_matrices = {}
for distance in distances:
    co_occurrence_matrices[distance] = interface.compute_co_occurrence_matrix(image, distance)

# Calculate Haralick features for each co-occurrence matrix
haralick_features = {}
for distance, co_occurrence_matrix in co_occurrence_matrices.items():
    features = {
        'Homogeneity': interface.calculate_homogeneity(co_occurrence_matrix),
        'Contrast': interface.calculate_contrast(co_occurrence_matrix),
        'Entropy': interface.calculate_entropy(co_occurrence_matrix)
    }
    haralick_features[distance] = features

# Print Haralick features
for distance, features in haralick_features.items():
    print(f"Haralick features for distance {distance}:\n{features}")