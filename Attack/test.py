import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import Normalize
from IFGSM import IFGSM
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]
)


def save_image(images, filenames, output_dir):
    """save high quality jpeg"""
    # mkdir(output_dir)
    for i, filename in enumerate(filenames):
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save(os.path.join(output_dir, filename))


def TNormalize(x, IsRe=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if not IsRe:
        x = Normalize(mean=mean, std=std)(x)
    elif IsRe:
        # tensor.shape:(3,w.h)
        for idx, i in enumerate(std):
            x[:, idx, :, :] *= i
        for index, j in enumerate(mean):
            x[:, index, :, :] += j
    return x


model = inception_v3(pretrained=True).eval().cuda()
dataset = ImageFolder(root="D:\data\ILSVRC2012_img_val", transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
success = 0
attack = IFGSM(steps=10, eps=4 / 255)
for batch, (images, labels) in enumerate(loader):
    images.cuda()
    labels.cuda()
    adv = attack.attack(model, images, labels)
    adv = TNormalize(adv, IsRe=True)
    save_image(adv, ["0.jpg"], "./Attack")
    break
