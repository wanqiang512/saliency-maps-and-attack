import os
import random
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch import tensor
from torchvision.transforms import Normalize
import torch.nn.functional as F

__all__ = ["RPA"]


class RPA:
    def __init__(
            self,
            ens: int = 60,
            iters: int = 10,
            eps: float = 16 / 255,
            alpha: float = 1.6 / 255,
            prob: float = 0.7,
            u: float = 1.0,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        self.ens = ens
        self.iters = iters
        self.eps = eps
        self.alpha = alpha
        self.prob = prob
        self.u = u
        self.device = device
        self.feature_map = {}
        self.weight = {}
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

    def save_fmaps(self, key: str) -> None:
        def forward_hook(model: nn.Module, input: tensor, output: tensor) -> None:
            self.feature_map[key] = output

        return forward_hook

    def save_weight(self, key: str) -> None:
        def backward_hook(model: nn.Module, grad_input: tensor, grad_output: tensor) -> None:
            self.weight[key] = grad_output[0]

        return backward_hook

    def patch_by_strides(self, image_shape: Optional[Tuple[int, ...]], patch_size: Optional[Tuple[int, ...]],
                         prob: float):
        mask = np.ones(image_shape)
        N0, H0, W0, C0 = mask.shape
        ph = H0 // patch_size[0]
        pw = W0 // patch_size[1]
        X = mask[:, :ph * patch_size[0], :pw * patch_size[1]]
        N, H, W, C = X.shape
        shape = (N, ph, pw, patch_size[0], patch_size[1], C)  # example shape == (10, 59, 59, 5, 5, 3)
        strides = (X.strides[0], X.strides[1] * patch_size[0], X.strides[2] * patch_size[0], *X.strides[1:])
        mask_patchs = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        mask_len = mask_patchs.shape[1] * mask_patchs.shape[2] * mask_patchs.shape[-1]
        ran_num = int(mask_len * (1 - prob))
        rand_list = np.random.choice(mask_len, ran_num, replace=False)  # Random sample P with Pm

        for i in range(mask_patchs.shape[1]):  # 使得mask_patchs: P prob 均匀分布
            for j in range(mask_patchs.shape[2]):
                for k in range(mask_patchs.shape[-1]):
                    if i * mask_patchs.shape[2] * mask_patchs.shape[-1] + j * mask_patchs.shape[-1] + k in rand_list:
                        mask_patchs[:, i, j, :, :, k] = np.random.uniform(0, 1, (
                            N, mask_patchs.shape[3], mask_patchs.shape[4]))

        img2 = np.concatenate(mask_patchs, axis=0, )
        img2 = np.concatenate(img2, axis=1)
        img2 = np.concatenate(img2, axis=1)
        img2 = img2.reshape((N, H, W, C))
        mask[:, :ph * patch_size[0], :pw * patch_size[1]] = img2
        return mask

    def get_RPA_loss(self, adv, model, layer, weight):
        self.feature_map.clear()
        self.weight.clear()
        logits = model(self.TNormalize(adv))
        attribution = self.feature_map[layer] * weight
        loss = torch.sum(attribution) / attribution.numel()
        return loss

    def __call__(self, model, inputs: tensor, labels: tensor, layer: str, *args, **kwargs):
        self.weight.clear()
        self.feature_map.clear()
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

        images = inputs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv = images.clone().detach()
        g = torch.zeros_like(images)
        shape = (images.shape[0], images.shape[2], images.shape[3], images.shape[1])  # N,H,W,C
        for i in range(self.iters):
            adv.requires_grad = True
            if i == 0:
                temp_weight = 0  # Aggregate gradient
                # path_size alternately sets 1，3，5，7
                for l in range(self.ens):
                    if l % 4 == 0:
                        mask1 = np.random.binomial(1, self.prob, size=shape)
                        mask2 = np.random.uniform(0, 1, size=shape)
                        mask = np.where(mask1 == 1, 1, mask2)
                    elif l % 4 == 1:
                        mask = self.patch_by_strides(shape, patch_size=(3, 3), prob=self.prob)
                    elif l % 4 == 2:
                        mask = self.patch_by_strides(shape, patch_size=(5, 5), prob=self.prob)
                    else:
                        mask = self.patch_by_strides(shape, patch_size=(7, 7), prob=self.prob)
                    mask = torch.from_numpy(mask).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2)
                    images_temp = (mask * images)
                    logits = model(self.TNormalize(images_temp))  # pytorch models normalized
                    logits = F.softmax(logits, dim=1)
                    one_hot = F.one_hot(labels, len(logits[0])).float()
                    score = one_hot * logits
                    loss = torch.sum(score)
                    loss.backward()
                    temp_weight += self.weight[layer]

                temp_weight.to(self.device)
                square = torch.sum(torch.square(temp_weight), [1, 2, 3], keepdim=True)
                weight = temp_weight / torch.sqrt(square)

            loss = self.get_RPA_loss(adv, model, layer, weight)
            grad = torch.autograd.grad(outputs=loss, inputs=adv, retain_graph=False)[0]
            g = self.u * g + grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            adv = adv - self.alpha * torch.sign(g)
            delta = torch.clip(adv - images, -self.eps, self.eps)
            adv = torch.clip(images + delta, 0, 1).detach_()
        return adv
