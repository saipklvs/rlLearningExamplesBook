import torch

if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device("cuda")
    print("Device: ", device)
    print("Device name: ", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")
    device = torch.device("cpu")
    print("Device: ", device)

x = torch.rand(5, 3).to(device)
y = torch.matmul(x, x.transpose(0, 1))

if device == torch.device("cuda"):
    print(x)
    print(y)

print("x: ", y)