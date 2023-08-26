import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchattacks
from torch import tensor
import random
from torchvision.transforms import Normalize

__all__ = ['AAAM']


class AAAM:
    def __init__(
            self,
            eps: float = 16 / 255,
            eta: float = 0.1,
            alpha: float = 1.6 / 255,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.eps = eps
        self.alpha = alpha
        self.device = device
        self.eta = eta
        self.criterion = nn.MSELoss(reduction='mean')
        self.seed_torch(1024)

    def seed_torch(self, seed):
        """Set a random seed to ensure that the results are reproducible"""
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    def TNormalize(self, x, IsRe, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        if not IsRe:
            x = Normalize(mean=mean, std=std)
        elif IsRe:
            # tensor.shape:(3,w.h)
            for idx, i in enumerate(std):
                x[:, idx, :, :] *= i
            for index, j in enumerate(mean):
                x[:, index, :, :] += j
        return x

    def LossCosine(self, I_y1: tensor, I_y2: tensor) -> tensor:
        """
        args:
        I_y1: 第一正确标签图像的SGLRP hot_map
        I_y2: 第二正确标签图像的SGLRP hot_map
        """
        temp = torch.dot(I_y1, I_y2)
        norm = torch.norm(I_y1) * torch.norm(I_y2)
        L_cosine = temp / norm
        L = -  torch.log((1 - L_cosine) / 2)
        return L

    def Loss(self, model: nn.Module, x: tensor, y: tensor, cam1: tensor, cam2: tensor) -> tensor:
        pred = model(x)
        loss = F.cross_entropy(pred, y)  # PGD loss
        gamma = 1000
        loss = self.LossCosine(cam1, cam2) - gamma * loss
        return loss

    @staticmethod
    def calculate_cam(self):
        pass

    def __call__(self, model, images, labels, *args, **kwargs):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv = images.clone().detach()
        n = 0

        while torch.sqrt(self.criterion(images, adv)) < self.eta:
            cam1 = ...
            cam2 = ...
            b, c, h, w = images.size()
            N = c * h * w  # batchsize = 1
            loss = self.Loss(model, images, labels, cam1, cam2)
            gt1 = torch.autograd.grad(loss, images, retain_graph=True)[0]
            gt1 = N * gt1 / torch.norm(gt1, p=1) + gt1 / torch.norm(gt1, p=2)
            gt1 = gt1 / 2
            # for i in range(4):  SI
            #     adv = adv / 2 ** i
            adv = adv - self.alpha * gt1
            n = n + 1
            delta = torch.clip(adv - images, -self.eps, self.eps)
            adv = torch.clip(images + delta, 0, 1)
        return adv
