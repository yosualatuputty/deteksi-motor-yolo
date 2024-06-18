import torch

# buat cek GPU (ga perlu)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Get CUDA version
cuda_version = torch.version.cuda

# Get the number of available GPUs
gpu_count = torch.cuda.device_count()

print(f"CUDA Available: {cuda_available}")
print(f"CUDA Version: {cuda_version}")
print(f"Number of GPUs: {gpu_count}")

if cuda_available:
    # Get the name of the GPU
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
