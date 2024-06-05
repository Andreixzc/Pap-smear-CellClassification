import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image

# Certifique-se de que os pesos da rede existem
effnet_binary_weights_path = '../Classificadores/ModelosTreinados/Effnet_Binary_Weights.pth'
effnet_multi_weights_path = '../Classificadores/ModelosTreinados/Effnet_Multi_Weights.pth'

if not (os.path.exists(effnet_binary_weights_path) and os.path.exists(effnet_multi_weights_path)):
    print("Os pesos da rede não foram encontrados.")
    exit()

# Carregando os pesos da rede
model_binary = models.efficientnet_b0(pretrained=False)
model_binary.classifier[1] = torch.nn.Linear(model_binary.classifier[1].in_features, 1)
model_binary.load_state_dict(torch.load(effnet_binary_weights_path, map_location=torch.device('cpu')))
model_binary.eval()

model_multi = models.efficientnet_b0(pretrained=False)
model_multi.classifier[1] = torch.nn.Linear(model_multi.classifier[1].in_features, 6)  # Número de classes no seu modelo
model_multi.load_state_dict(torch.load(effnet_multi_weights_path, map_location=torch.device('cpu')))
model_multi.eval()

# Função para pré-processar a imagem
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

# Função para fazer previsões
def predict(image_path, binary_threshold=0.5):
    # Pré-processa a imagem
    image = preprocess_image(image_path)

    # Previsão do modelo binário
    with torch.no_grad():
        prediction_binary = torch.sigmoid(model_binary(image)).item()
        # Converte a probabilidade em uma previsão binária
        prediction_binary = 1 if prediction_binary >= binary_threshold else 0

    # Previsão do modelo multiclasse
    with torch.no_grad():
        output_multi = model_multi(image)
        prediction_multi = torch.argmax(output_multi, dim=1).item()

    return prediction_binary, prediction_multi

# Caminho para a imagem
image_path = '5.png'

# Fazendo previsões
prediction_binary, prediction_multi = predict(image_path)

# Imprimindo as previsões
print("Previsão do Modelo Binário:", prediction_binary)
print("Previsão do Modelo Multiclasse:", prediction_multi)
