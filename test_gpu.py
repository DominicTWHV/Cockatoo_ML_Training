import torch

# validate if GPU is available and print info
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")