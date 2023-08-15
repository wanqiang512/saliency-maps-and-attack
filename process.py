import torch
import torchattacks
import torchvision.transforms as transforms

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]


trans = transforms.Normalize(mean, std)


def denormalize(tens, mean=mean, std=std):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]


def AdvUpdate(images, adv, pert, eps):
    images = images.clone().detach()
    adv = adv.clone().detach()
    adv = denormalize(adv)
    images = denormalize(images)
    adv = adv + pert
    diff = adv - images
    delta = torch.clip(diff, -eps, eps)
    adv = torch.clip(images + delta, 0, 1)
    adv = trans(adv)
    return adv.clone().detach()
