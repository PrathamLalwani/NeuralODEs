import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Check number of GPUs
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Get current device
print(f"Current device: {torch.cuda.current_device()}")

# Get device name
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Test device usage
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = x + y
    print(f"Computation successful on: {z.device}")
else:
    print("CUDA still not available!")