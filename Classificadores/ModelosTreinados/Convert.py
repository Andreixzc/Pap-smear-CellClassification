import os
import xgboost as xgb
import pickle
import torch
import torch.nn as nn
from torchvision import models

# Create the directory for the converted models if it doesn't exist
os.makedirs('convertedModels', exist_ok=True)

# Convert XGBoost models to JSON format
def convert_xgb_model(input_path, output_path):
    with open(input_path, 'rb') as f:
        model = pickle.load(f)
    model.save_model(output_path)

# Convert XGBoost models and save to 'convertedModels' folder
convert_xgb_model('xgboostMulti_model.pkl', 'convertedModels/xgboostMulti_model.json')
convert_xgb_model('xgboostBinary_model.pkl', 'convertedModels/xgboostBinary_model.json')

# Function to convert and save EfficientNet model
def convert_effnet_model(input_path, output_path, num_classes, binary=False):
    # Load the model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Modify the classifier to match the number of classes
    if binary:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    else:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Load the trained weights
    model.load_state_dict(torch.load(input_path))
    model.eval()

    # Create example input tensor
    example_input = torch.randn(1, 3, 224, 224)

    # Export the model to TorchScript
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(output_path)

# Convert and save binary classification model to 'convertedModels' folder
convert_effnet_model('Effnet_Binary_Weights.pth', 'convertedModels/Effnet_Binary_Weights.pt', num_classes=1, binary=True)

# Convert and save multi-class classification model to 'convertedModels' folder
# Specify the number of classes in your multi-class problem
convert_effnet_model('Effnet_Multi_Weights.pth', 'convertedModels/Effnet_Multi_Weights.pt', num_classes=6, binary=False)
