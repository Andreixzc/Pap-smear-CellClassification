import numpy as np

def compute_co_occurrence_matrix(img, distance):
    co_occurrence_matrix = np.zeros((16, 16), dtype=np.int32)

    limit = img.shape[1] - distance
    for y in range(img.shape[0]):
        for x in range(limit):
            g1 = img[y, x] // (256 // 16)
            g2 = img[y, x + distance] // (256 // 16)
            co_occurrence_matrix[g1, g2] += 1

    return co_occurrence_matrix

def calculate_homogeneity(co_matrix):
    homogeneity = 0.0

    size = co_matrix.shape[0]
    for i in range(size):
        for j in range(size):
            homogeneity += co_matrix[i, j] / (1.0 + abs(i - j))

    return homogeneity

def calculate_contrast(co_matrix):
    contrast = 0.0

    size = co_matrix.shape[0]
    for i in range(size):
        for j in range(size):
            contrast += (i - j) * (i - j) * co_matrix[i, j]

    return contrast

def calculate_entropy(co_matrix):
    entropy = 0.0

    size = co_matrix.shape[0]
    for i in range(size):
        for j in range(size):
            value = co_matrix[i, j]
            if value > 0:
                entropy -= value * np.log2(value)

    return entropy



import cv2
import numpy as np
# Load a sample grayscale image
image_path = '5.png'  # Replace with the path to your sample image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Specify distances
distances = [1, 2, 4, 8, 16, 32]

# Calculate co-occurrence matrices
co_occurrence_matrices = {}
for distance in distances:
    co_occurrence_matrices[distance] = compute_co_occurrence_matrix(image, distance)

# Calculate Haralick features for each co-occurrence matrix
haralick_features = {}
for distance, co_occurrence_matrix in co_occurrence_matrices.items():
    features = {
        'Homogeneity': calculate_homogeneity(co_occurrence_matrix),
        'Contrast': calculate_contrast(co_occurrence_matrix),
        'Entropy': calculate_entropy(co_occurrence_matrix)
    }
    haralick_features[distance] = features

# Print Haralick features
for distance, features in haralick_features.items():
    print(f"Haralick features for distance {distance}:\n{features}")