import torch
import torchvision
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
print(torch.__version__)
print(torchvision.__version__)