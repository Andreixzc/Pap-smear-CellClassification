import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image
import numpy as np
from joblib import load

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

        
    @staticmethod
    def grayHistogram(image):
        # Verificar se a imagem está em BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray_image = image  # Já está em tons de cinza
        else:
            raise ValueError("Imagem com número de canais não suportado")

        hist = cv2.calcHist([gray_image], [0], None, [16], [0, 256])
        hist = hist.flatten()
        return hist
    

    @staticmethod
    def colorHistogram(image):
         # Verificar se a imagem está em BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError("A imagem precisa ter 3 canais (BGR)")
        # Converter a imagem de BGR para HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Definir o número de bins para H e V
        h_bins = 16
        v_bins = 8

        # Definir os intervalos para H e V
        h_range = [0, 180]
        v_range = [0, 256]

        # Calcular o histograma 2D
        hist = cv2.calcHist([hsv_image], [0, 2], None, [h_bins, v_bins], h_range + v_range)

        # Normalizar o histograma para facilitar a visualização
        cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        return hist


    @staticmethod
    def preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  # Adiciona uma dimensão extra para batch
        return image

    @staticmethod
    def predict(imagemOriginal,image):
        #imagemcv2 e path da imagem
        # Extração dos momentos de Hu
        hu_moments = InterfaceUtils.extract_hu_moments(imagemOriginal)
        entrada = np.array(hu_moments).reshape(1, -1)
    
        # Caminhos dos modelos pré-treinados
        binary_model_path = '../Classificadores/ModelosTreinados/xgboostBinary_model.pkl'
        multi_model_path = '../Classificadores/ModelosTreinados/xgboostMulti_model.pkl'
        effnet_binary_weights_path = '../Classificadores/ModelosTreinados/Effnet_Binary_Weights.pth'
        effnet_multi_weights_path = '../Classificadores/ModelosTreinados/Effnet_Multi_Weights.pth'

        # Carregar modelos binário e multiclasse
        model_binario = load(binary_model_path)
        model_multiclasse = load(multi_model_path)

        # Previsões usando os modelos binário e multiclasse
        previsao_binario = model_binario.predict(entrada)[0]
        previsao_multiclasse = model_multiclasse.predict(entrada)[0]

        # Carregar os pesos da rede neural
        #investigar parametro pre-trained = false.
        model_binary = models.efficientnet_b0()
        model_binary.classifier[1] = torch.nn.Linear(model_binary.classifier[1].in_features, 1)
        model_binary.load_state_dict(torch.load(effnet_binary_weights_path, map_location=torch.device('cpu')))
        model_binary.eval()

        model_multi = models.efficientnet_b0()
        model_multi.classifier[1] = torch.nn.Linear(model_multi.classifier[1].in_features, 6)  # Número de classes no seu modelo
        model_multi.load_state_dict(torch.load(effnet_multi_weights_path, map_location=torch.device('cpu')))
        model_multi.eval()

        # Pré-processamento da imagem para modelos de rede neural
        preprocessed_image = InterfaceUtils.preprocess_image(image)

        # Previsão do modelo binário
        with torch.no_grad():
            prediction_binary = torch.sigmoid(model_binary(preprocessed_image)).item()
            # Converte a probabilidade em uma previsão binária
            prediction_binary = 1 if prediction_binary >= 0.5 else 0

        # Previsão do modelo multiclasse
        with torch.no_grad():
            output_multi = model_multi(preprocessed_image)
            prediction_multi = torch.argmax(output_multi, dim=1).item()

        
        return {
            "previsao_binario_modelo_xgboost": previsao_binario,
            "previsao_multiclasse_modelo_xgboost": previsao_multiclasse,
            "previsao_binario_modelo_effnet": prediction_binary,
            "previsao_multiclasse_modelo_effnet": prediction_multi
        }

    @staticmethod
    def predict2(imagemOriginal, image):
        # Extract Hu moments
        hu_moments = InterfaceUtils.extract_hu_moments(imagemOriginal)
        entrada = np.array(hu_moments).reshape(1, -1)

        # Paths to pre-trained models
        binary_model_path = '../Classificadores/ModelosTreinados/xgboostBinary_model.pkl'
        multi_model_path = '../Classificadores/ModelosTreinados/xgboostMulti_model.pkl'
        effnet_binary_weights_path = '../Classificadores/ModelosTreinados/Effnet_Binary_Weights.pth'
        effnet_multi_weights_path = '../Classificadores/ModelosTreinados/Effnet_Multi_Weights.pth'

        # Load binary and multi-class models
        model_binario = load(binary_model_path)
        model_multiclasse = load(multi_model_path)

        # Predictions using the binary and multi-class models
        previsao_binario = model_binario.predict(entrada)[0]
        previsao_multiclasse = model_multiclasse.predict(entrada)[0]

        # Load neural network weights
        model_binary = models.efficientnet_b0(pretrained=False)
        model_binary.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(model_binary.classifier[1].in_features, 1)
        )
        state_dict = torch.load(effnet_binary_weights_path, map_location=torch.device('cpu'))
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("classifier.1.1", "classifier.1")
            new_state_dict[new_key] = value
        model_binary.load_state_dict(new_state_dict)
        model_binary.eval()

        model_multi = models.efficientnet_b0(pretrained=False)
        model_multi.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(model_multi.classifier[1].in_features, 6)  # Number of classes in your model
        )
        state_dict = torch.load(effnet_multi_weights_path, map_location=torch.device('cpu'))
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("classifier.1.1", "classifier.1")
            new_state_dict[new_key] = value
        model_multi.load_state_dict(new_state_dict)
        model_multi.eval()

        # Preprocess the image for neural network models
        preprocessed_image = InterfaceUtils.preprocess_image(image)

        # Binary model prediction
        with torch.no_grad():
            prediction_binary = torch.sigmoid(model_binary(preprocessed_image)).item()
            # Convert the probability into a binary prediction
            prediction_binary = 1 if prediction_binary >= 0.5 else 0

        # Multi-class model prediction
        with torch.no_grad():
            output_multi = model_multi(preprocessed_image)
            prediction_multi = torch.argmax(output_multi, dim=1).item()

        return {
            "previsao_binario_modelo_xgboost": previsao_binario,
            "previsao_multiclasse_modelo_xgboost": previsao_multiclasse,
            "previsao_binario_modelo_effnet": prediction_binary,
            "previsao_multiclasse_modelo_effnet": prediction_multi
        }



