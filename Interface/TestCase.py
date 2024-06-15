import cv2
from matplotlib import pyplot as plt
from InterfaceUtils import InterfaceUtils
interface = InterfaceUtils()

imagemOriginal = cv2.imread("5.png")
imagemOriginalPath = "5.png"
imagemCinza = cv2.imread("5.png", cv2.IMREAD_GRAYSCALE)

print("Momentos de Hu da imagem????:")
print(interface.extract_hu_moments(imagemOriginal))



# Distancias da matriz
distances = [1, 2, 4, 8, 16, 32]

# Calculando matriz
co_occurrence_matrices = {}
for distance in distances:
    co_occurrence_matrices[distance] = interface.compute_co_occurrence_matrix(imagemCinza, distance)

# Calculando Descritores de homogeniedade, contraste e entropia pra cada uma das matrizes.
#
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

# OBS: Tem que calcular a matriz antes pra depois calcular os descritores. Não sei ainda se esse é o jeito certo de calcular, porque 
# o gpt fez dois códigos. Mas dessa forma ai ficou igual a do joão, qualquer coisa mudamos pro jeito anterior que ta no 'Corrency2.py'.3





# Gerar o histograma 2D
histogramaColorido2D = interface.colorHistogram(imagemOriginal)

# Plotar o histograma 2D
plt.imshow(histogramaColorido2D, interpolation='nearest', aspect='auto')
plt.title('Histograma 2D para H e V')
plt.xlabel('Valores de V')
plt.ylabel('Valores de H')
plt.colorbar()
plt.show()

#Histograma cinza:
histogramaCinza = interface.grayHistogram(imagemCinza)
plt.figure()
plt.title("Histograma em Tons de Cinza")
plt.xlabel("Intensidade")
plt.ylabel("Número de Pixels")
plt.plot(histogramaCinza)
plt.xlim([0, 256])
plt.show()


#predict
print(interface.predict(imagemOriginal,imagemOriginalPath))