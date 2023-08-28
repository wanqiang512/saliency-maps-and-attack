import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import Normalize

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
]
)

model = inception_v3(pretrained=True).eval()
dataset = ImageFolder(root="D:\data\ILSVRC2012_img_val", transform=transform)
loader = DataLoader(dataset, batch_size=100, shuffle=True)
success = 0
for batch, (images, labels) in enumerate(loader):
    print(images)
    break
