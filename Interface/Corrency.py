import cv2
import numpy as np

def calculate_co_occurrence_matrix(image, distance):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate co-occurrence matrix
    co_occurrence_matrix = np.zeros((256, 256))
    
    for i in range(len(gray_image)):
        for j in range(len(gray_image[0])):
            if j + distance < len(gray_image[0]):
                co_occurrence_matrix[gray_image[i][j]][gray_image[i][j + distance]] += 1
    
    # Normalize the co-occurrence matrix
    co_occurrence_matrix /= np.sum(co_occurrence_matrix)
    
    return co_occurrence_matrix

def calculate_haralick_features(co_occurrence_matrix):
    # Calculate Haralick features
    features = {}
    # Entropy
    entropy = -np.sum(np.multiply(co_occurrence_matrix, np.log2(co_occurrence_matrix + (co_occurrence_matrix == 0))))
    features['Entropy'] = entropy

    # Homogeneity
    homogeneity = np.sum(np.divide(1, 1 + np.square(np.arange(co_occurrence_matrix.shape[0]) - np.arange(co_occurrence_matrix.shape[1])))*co_occurrence_matrix)
    features['Homogeneity'] = homogeneity
    
    # Contrast
    contrast = np.sum(np.multiply(np.square(np.arange(co_occurrence_matrix.shape[0]) - np.arange(co_occurrence_matrix.shape[1])), co_occurrence_matrix))
    features['Contrast'] = contrast
    
    
    
    return features

# Load the image
image = cv2.imread('5.png')  # Replace 'your_image.jpg' with the path to your image

# Specify distances
distances = [1, 2, 4, 8, 16, 32]

# Calculate co-occurrence matrices
co_occurrence_matrices = {}
for distance in distances:
    co_occurrence_matrices[distance] = calculate_co_occurrence_matrix(image, distance)

# Calculate Haralick features for each co-occurrence matrix
haralick_features = {}
for distance, co_occurrence_matrix in co_occurrence_matrices.items():
    haralick_features[distance] = calculate_haralick_features(co_occurrence_matrix)

# Print Haralick features
for distance, features in haralick_features.items():
    print(f"Haralick features for distance {distance}:\n{features}")
