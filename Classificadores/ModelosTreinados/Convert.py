import xgboost as xgb
import pickle
import torch
from efficientnet_pytorch import EfficientNet

# Convert XGBoost models to JSON format
def convert_xgb_model(input_path, output_path):
    with open(input_path, 'rb') as f:
        model = pickle.load(f)
    model.save_model(output_path)

# Convert XGBoost models
convert_xgb_model('xgboostMulti_model.pkl', 'xgboostMulti_model.json')
convert_xgb_model('xgboostBinary_model.pkl', 'xgboostBinary_model.json')

# Convert PyTorch models to TorchScript format
def convert_effnet_model(input_path, output_path):
    # Load the model
    model = EfficientNet.from_name('efficientnet-b0')
    model.load_state_dict(torch.load(input_path))
    model.eval()

    # Create example input tensor
    example_input = torch.randn(1, 3, 224, 224)

    # Export the model to TorchScript
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(output_path)

# Convert and save binary classification model
convert_effnet_model('Effnet_Binary_Weights.pth', 'Effnet_Binary_Weights.pt')

# Convert and save multi-class classification model
convert_effnet_model('Effnet_Multi_Weights.pth', 'Effnet_Multi_Weights.pt')
