import torch

cuda_disponivel = torch.cuda.is_available()
print(f"CUDA disponível: {cuda_disponivel}")