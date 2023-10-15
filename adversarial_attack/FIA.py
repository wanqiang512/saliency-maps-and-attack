import os
import random
import torch
import torch.nn as nn
import numpy as np
from torch import tensor
from torchvision.transforms import Normalize
import torch.nn.functional as F
__all__ = ['FIA']


class FIA:
    def __init__(
            self,
            eps=16 / 255,
            steps=10,
            u=1,
            a=1.6 / 255,
            ens=30,
            drop_pb=0.7,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            seed=1234
    ):
        self.eps = eps
        self.device = device
        self.steps = steps
        self.u = u
        self.a = a
        self.ens = ens
        self.drop_pb = drop_pb
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

    def get_FIA_loss(self, x, model, weight, layer):
        self.feature_map.clear()
        loss = 0
        logits = model(self.TNormalize(x))
        attribution = self.feature_map[layer] * weight
        loss = torch.sum(attribution) / attribution.numel()
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

        a = self.a
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
                    mask = np.random.binomial(1, self.drop_pb, size=adv.shape)
                    mask = torch.from_numpy(mask).to(self.device)
                    image_tmp = images.clone() * mask
                    logits = model(self.TNormalize(image_tmp))
                    logits = F.softmax(logits, 1)
                    labels_onehot = F.one_hot(labels, len(logits[0])).float()
                    score = logits * labels_onehot
                    loss = torch.sum(score)
                    loss.backward()
                    temp_weight += self.weight[layer].clone().detach()

                temp_weight.to(self.device)
                square = torch.sum(torch.square(temp_weight), [1, 2, 3], keepdim=True)
                weight = temp_weight / torch.sqrt(square)

            loss = self.get_FIA_loss(adv, model, weight, layer)
            loss.backward()
            adv_grad = adv.grad.clone()
            adv.grad.data.zero_()
            g = self.u * g + (adv_grad / (torch.mean(torch.abs(adv_grad), [1, 2, 3], keepdim=True)))
            adv = adv.detach_() - a * torch.sign(g)
            delta = torch.clip(adv - inputs, -self.eps, self.eps)
            adv = torch.clip(inputs + delta, 0, 1).detach_()
        return adv
