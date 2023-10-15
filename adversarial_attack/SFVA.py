import os
import random
import torch
import torch.nn as nn
import numpy as np
from torch import tensor
from torchvision.transforms import Normalize
import torch.nn.functional as F
from typing import List, Optional

__all__ = ['SFVA']


class SFVA:
    def __init__(
            self,
            a=0.5,
            eps=16 / 255,
            steps=10,
            u=1,
            alpha=1.6 / 255,
            ens=30,
            prob=0.9,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            seed=1234,
    ):
        self.a = a
        self.eps = eps
        self.device = device
        self.steps = steps
        self.u = u
        self.alpha = alpha
        self.ens = ens
        self.prob = prob
        self.feature_map = {}
        self.weight = {}
        self.seed_torch(seed)

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

    def advanced_mask(self, img, p):
        # get width and height of the image
        s = img.shape
        b = s[0]
        wd = s[1]
        ht = s[2]
        img_copy = np.copy(img)
        np.random.shuffle(img_copy)
        # possible grid size, 0 means no hiding
        grid_sizes = [0, 20, 40, 60, 80]
        # hiding probability
        hide_prob = p
        # randomly choose one grid size
        grid_size = grid_sizes[random.randint(0, len(grid_sizes) - 1)]
        # hide the patches
        if grid_size != 0:
            for x in range(0, wd, grid_size):
                for y in range(0, ht, grid_size):
                    x_end = min(wd, x + grid_size)
                    y_end = min(ht, y + grid_size)
                    if random.random() <= hide_prob:
                        img[:, x:x_end, y:y_end, :] = np.random.uniform(low=0, high=1,
                                                                        size=np.shape(img[:, x:x_end, y:y_end, :]))
        return img

    def get_SFVA_loss(self, inputs, adv, model, weight, layers: Optional[List]):
        loss = 0
        for layer in layers:
            self.feature_map.clear()
            self.weight.clear()
            logits = model(self.TNormalize(inputs))
            clean_attribution = self.feature_map[layer] * weight
            self.feature_map.clear()
            self.weight.clear()
            logits = model(self.TNormalize(adv))
            adv_attribution = self.feature_map[layer] * weight
            blank = torch.zeros_like(clean_attribution)
            pclean = torch.where(clean_attribution >= 0, clean_attribution, blank)
            padv = torch.where(adv_attribution >= 0, adv_attribution, blank)
            nclean = torch.where(clean_attribution < 0, clean_attribution, blank)
            nadv = torch.where(clean_attribution < 0, adv_attribution, blank)
            p_attribuion = pclean - padv
            n_attribution = nclean - nadv
            loss += (torch.sum(p_attribuion) + torch.sum(n_attribution)) / clean_attribution.numel()
        loss = loss / len(layers)
        return loss

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

    def save_fmaps(self, key: str) -> None:
        def forward_hook(model: nn.Module, input: tensor, output: tensor) -> None:
            self.feature_map[key] = output

        return forward_hook

    def save_weight(self, key: str) -> None:
        def backward_hook(model: nn.Module, grad_input: tensor, grad_output: tensor) -> None:
            self.weight[key] = grad_output[0]

        return backward_hook

    def __call__(self, model: nn.Module, inputs: torch.tensor, labels: torch.tensor, layer: str):
        self.feature_map.clear()
        self.weight.clear()
        if torch.max(inputs) > 1 or torch.min(inputs) < 0:
            raise ValueError('Input must have a range [0, 1] (max: {}, min: {})'.format(
                torch.max(inputs), torch.min(inputs))
            )

        submodule_dict = dict(model.named_modules())
        try:
            submodule_dict[layer].register_backward_hook(self.save_weight(layer))
            submodule_dict[layer].register_forward_hook(self.save_fmaps(layer))
        except Exception as e:
            raise e(f'can not find {layer} in {model.__class__.__name__}')

        g = torch.zeros_like(inputs)
        images = inputs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv = images.clone().detach()
        for i in range(self.steps):
            adv.requires_grad = True
            if i == 0:
                temp_weight = 0
                for l in range(self.ens):
                    self.weight.clear()
                    self.feature_map.clear()
                    temp = inputs.clone().detach()
                    temp = temp.cpu().numpy().transpose(0, 2, 3, 1)
                    images_tmp = self.advanced_mask(temp, 0.1)
                    images_tmp = torch.from_numpy(images_tmp).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32)
                    images_tmp = self.TNormalize(images_tmp)
                    images_tmp += torch.from_numpy(
                        np.random.uniform(low=-self.a, high=self.a, size=images_tmp.shape)).to(self.device)
                    images_tmp = images_tmp * (1 - l / self.ens)
                    # import matplotlib.pyplot as plt
                    # plt.imshow(images_tmp.detach().cpu().squeeze().numpy().transpose(1, 2, 0))
                    # plt.show()
                    logits = model(images_tmp)
                    logits = F.softmax(logits, 1)
                    labels_onehot = F.one_hot(labels, len(logits[0])).float()
                    score = logits * labels_onehot
                    loss = torch.sum(score)
                    loss.backward()
                    temp_weight += self.weight[layer]

                temp_weight.to(self.device)
                square = torch.sum(torch.square(temp_weight), [1, 2, 3], keepdim=True)
                weight = temp_weight / torch.sqrt(square)

            loss = self.get_SFVA_loss(inputs, adv, model, weight, [layer])
            loss.backward()
            adv_grad = adv.grad.clone()
            adv.grad.data.zero_()
            g = self.u * g + (adv_grad / (torch.mean(torch.abs(adv_grad), [1, 2, 3], keepdim=True)))
            adv = adv - self.alpha * torch.sign(g)
            delta = torch.clip(adv - inputs, -self.eps, self.eps)
            adv = torch.clip(inputs + delta, 0, 1).detach_()
        return adv
