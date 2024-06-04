import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Certifique-se de que a pasta 'Desempenho' exista
os.makedirs('Desempenho', exist_ok=True)

# Defina seu próprio Dataset para carregar as imagens e labels a partir do CSV
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(self.data['Label'].unique())}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_path = self.data.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx, 1]
        label = self.label_to_idx[label]  # Converte o label para um índice numérico

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label).long()  # Converte o label para tensor

        return image, label

# Caminho para o arquivo CSV
csv_file = '../../../Csvs/image_paths_labelsFix.csv'

# Defina as transformações para o pré-processamento das imagens
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Carregue o dataset
dataset = CustomDataset(csv_file, transform=transform)

# Divida o dataset em conjuntos de treino e teste
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Crie DataLoaders para treino e teste
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Carregue o modelo EfficientNet pré-treinado
model = models.efficientnet_b0(pretrained=True)

# Modifique a camada final do modelo para se ajustar ao número de classes no seu dataset
num_classes = len(set(dataset.data.iloc[:, 1]))  # Calcula o número de classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Defina o dispositivo para treinamento (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Defina o otimizador e a função de perda
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Função para treinar o modelo
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, save_path='Effnet_Multi_Weights.pth'):
    train_acc_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc_train = correct_train / total_train
        train_acc_history.append(epoch_acc_train)

        # Avalia no conjunto de teste
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        epoch_acc_test = correct_test / total_test
        test_acc_history.append(epoch_acc_test)


        # save the print bellow on a txt file:
        with open('Desempenho/epoch_results.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc_train:.4f}, Test Acc: {epoch_acc_test:.4f}\n')
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc_train:.4f}, Test Acc: {epoch_acc_test:.4f}')

    # Salvar o modelo treinado
    torch.save(model.state_dict(), save_path)
    print(f'Model weights saved to {save_path}')

    return train_acc_history, test_acc_history

# Treine o modelo
train_acc_history, test_acc_history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, save_path='../../ModelosTreinados/Effnet_Multi_Weights.pth')

# Função para avaliar o modelo e plotar a matriz de confusão
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

    # Plotar a matriz de confusão
    cm = confusion_matrix(all_labels, all_predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(dataset.label_to_idx.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('Desempenho/confusion_matrix.png')
    plt.close()
    # plt.show()

# Função para plotar os gráficos de aprendizagem
def plot_learning_curves(train_acc_history, test_acc_history):
    epochs = range(1, len(train_acc_history) + 1)
    plt.plot(epochs, train_acc_history, 'bo-', label='Training accuracy')
    plt.plot(epochs, test_acc_history, 'ro-', label='Testing accuracy')
    plt.title('Training and Testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Desempenho/train_test_acc.png')
    plt.close()
    # plt.show()

# Avalie o modelo
evaluate_model(model, test_loader)

# Plote os gráficos de aprendizagem
plot_learning_curves(train_acc_history, test_acc_history)
