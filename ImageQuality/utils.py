import numpy as np
import torch
from torch.autograd import Variable

from ImageQuality import pytorch_ssim


# PSNR
def PSNR(img1, img2):  # 上升
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# MSE
def MSE(img1, img2):  # 下降
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


# SSIM
def SSIM(img1, img2):  # 上升
    img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0) / 255.0
    img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0) / 255.0
    img1 = Variable(img1, requires_grad=False)  # torch.Size([256, 256, 3])
    img2 = Variable(img2, requires_grad=False)
    ssim_value = pytorch_ssim.ssim(img1, img2).item()
    return ssim_value
