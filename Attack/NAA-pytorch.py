import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms import Normalize


class NAA:
    def __init__(self, esp=16 / 255, steps=10, u=1, a=1.6 / 255, ens=30,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.esp = esp
        self.device = device
        self.steps = steps
        self.u = u
        self.a = a
        self.ens = ens
        self.feature_map = []
        self.weight = []
        self.seed_torch(1024)

    def get_NAA_loss(self, x, model, weight, base_feature):
        self.feature_map.clear()
        gamma = 1.0
        logits = model(self.TNormalize(x))
        attribution = (self.feature_map[0] - base_feature) * weight
        blank = torch.zeros_like(attribution)
        positive = torch.where(attribution >= 0, attribution, blank)
        negative = torch.where(attribution < 0, attribution, blank)
        positive = positive
        negative = negative
        balance_attribution = positive + gamma * negative
        loss = torch.sum(balance_attribution) / balance_attribution.numel()
        return loss

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

    def TNormalize(self, x, IsRe=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        if not IsRe:
            x = Normalize(mean=mean, std=std)(x)
        elif IsRe:
            # tensor.shape:(3,w.h)
            for idx, i in enumerate(std):
                x[:, idx, :, :] *= i
            for index, j in enumerate(mean):
                x[:, index, :, :] += j
        return x

    def forward_hook(self, model, input, output):
        self.feature_map.append(output)

    def backward_hook(self, model, grad_input, grad_output):
        self.weight.append(grad_output[0])

    def __call__(self, model, inputs, labels, layer):
        if torch.max(inputs) > 1 or torch.min(inputs) < 0:
            raise ValueError('Input must have a range [0, 1] (max: {}, min: {})'.format(
                torch.max(inputs), torch.min(inputs))
            )
        self.weight.clear()
        self.feature_map.clear()
        if hasattr(model, layer):
            getattr(model, layer).register_backward_hook(self.backward_hook)
            getattr(model, layer).register_forward_hook(self.forward_hook)
        else:
            raise Exception(f'can not find {layer} in {model.__class__.__name__}')

        a = self.a
        g = torch.zeros_like(inputs)
        images = inputs.clone().detach()
        labels = labels.clone().detach()
        adv = images.clone().detach()

        for i in range(self.steps):
            adv.requires_grad = True
            if i == 0:
                temp_weight = 0
                self.weight.clear()
                for l in range(self.ens):
                    self.feature_map.clear()
                    x_base = torch.zeros_like(inputs)
                    temp_noise = np.random.normal(size=inputs.shape, loc=0.0, scale=0.2)
                    temp_noise = torch.from_numpy(temp_noise).to(self.device, dtype=torch.float32)
                    image_tmp = torch.clip(inputs.clone() + temp_noise, 0, 1)
                    image_tmp = (image_tmp * (1 - l / self.ens) + (l / self.ens) * x_base)
                    logits = model(self.TNormalize(image_tmp))
                    logits = nn.functional.softmax(logits, 1)
                    labels_onehot = nn.functional.one_hot(labels, num_classes=len(logits[0])).float()
                    score = logits * labels_onehot
                    loss = torch.sum(score)
                    loss.backward()
                for w in self.weight:
                    temp_weight += w

                weight = temp_weight.to(self.device).clone().detach()
                square = torch.sum(torch.square(weight), [1, 2, 3], keepdim=True)
                weight = -weight / torch.sqrt(square)

            self.feature_map.clear()
            base_line = torch.zeros_like(inputs)
            logits = model(self.TNormalize(base_line))
            base_feature = self.feature_map[0]
            self.feature_map.clear()
            loss = self.get_NAA_loss(adv, model, weight, base_feature)
            loss.backward()
            adv_grad = adv.grad.clone()
            adv.grad.data.zero_()
            g = self.u * g + (adv_grad / (torch.mean(torch.abs(adv_grad), [1, 2, 3], keepdim=True)))
            adv = adv + a * torch.sign(g)
            diff = adv - inputs
            noise = torch.clip(diff, -self.esp, self.esp)
            adv = torch.clip(adv + noise, 0, 1).detach_()
        return adv
