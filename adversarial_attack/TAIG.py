import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Normalize

__all__ = ['TAIG']


# model must be nn.Sequential {Normalized(mean,std), model}
class TAIG:
    def __init__(
            self,
            eps: float = 16 / 255,
            ens: int = 20,
            alpha: float = 1.6 / 255,
            R: bool = True,
            iters: int = 10,
            device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            u=1,
    ):
        """TAIG-S and TAIG-R
        >>> attack = TAIG(...)
        >>> adv = attack(...)
        """
        self.u = u
        self.eps = eps
        self.ens = ens
        self.alpha = alpha
        self.R = R
        self.iters = iters
        self.device = device
        self.seed_torch(1234)

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

    def TNormalize(self, x, IsRe=False, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        if not IsRe:
            x = Normalize(mean=mean, std=std)(x)
        elif IsRe:
            # tensor.shape:(3,w.h)
            for idx, i in enumerate(std):
                x[:, idx, :, :] *= i
            for index, j in enumerate(mean):
                x[:, index, :, :] += j
        return x

    def compute_ig(self, model, images, labels):
        baseline = torch.zeros_like(images)
        scaled_inputs = [baseline + (float(i) / self.ens) * (images - baseline) for i in range(0, self.ens + 1)]
        scaled_inputs = torch.stack(scaled_inputs).to(self.device, dtype=torch.float32)
        if self.R:
            temp = np.random.uniform(-self.eps, self.eps, scaled_inputs.shape)
            temp = torch.from_numpy(temp).to(self.device, dtype=torch.float32)
            scaled_inputs = scaled_inputs + temp
        IG = []
        for _ in range(scaled_inputs.shape[1]):
            temp_label = labels[_].unsqueeze(0)
            temp_image = scaled_inputs[:, _, :, :, :].clone().detach()
            temp_image.requires_grad = True
            logits = model(self.TNormalize(temp_image))
            score = logits[:, temp_label]
            loss = torch.mean(score)
            model.zero_grad()
            grad = torch.autograd.grad(loss, temp_image, retain_graph=False, create_graph=False)[0]
            avg_grad = torch.mean(grad, dim=0)
            IG.append((temp_image[-1] - temp_image[0]) * avg_grad)
        IG = torch.stack(IG).to(self.device)
        return IG

    def __call__(self, model, images, labels, *args, **kwargs):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv = images.clone().detach()
        g = torch.zeros_like(images)

        for i in range(self.iters):
            ig = self.compute_ig(model, adv, labels)
            g = self.u * g + ig / torch.mean(torch.abs(ig), dim=(1, 2, 3), keepdim=True)
            adv = adv.detach() - self.alpha * torch.sign(g)
            adv = torch.where(adv > images + self.eps, images + self.eps, adv)
            adv = torch.where(adv < images - self.eps, images - self.eps, adv)
            adv = torch.clip(adv, 0, 1)

        return adv
