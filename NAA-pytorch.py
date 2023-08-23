import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import copy


def Re_transforms(tensor, mean: list, std: list):
    # tensor.shape:(3,w.h)
    for idx, i in enumerate(std):
        tensor[:, idx, :, :] *= i
    for index, j in enumerate(mean):
        tensor[:, index, :, :] += j
    return tensor


class NAA:
    def __init__(self, esp=8 / 255, steps=10, u=1, a=1.6 / 255, ens=30,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.esp = esp
        self.device = device
        self.steps = steps
        self.u = u
        self.a = a
        self.ens = ens
        self.feature_map = []
        self.weight = []

    def get_NAA_loss(self, x, models, weights, base_feature):
        self.feature_map.clear()
        loss = 0
        gamma = 1.0
        for idx, model in enumerate(models):
            logits = model(x)
            attribution = (self.feature_map[idx] - base_feature[idx]) * weights[idx]
            blank = torch.zeros_like(attribution)
            positive = torch.where(attribution >= 0, attribution, blank)
            negative = torch.where(attribution < 0, attribution, blank)
            positive = positive
            negative = negative
            balance_attribution = positive + gamma * negative
            loss += torch.sum(balance_attribution) / balance_attribution.numel()
        return loss

    def forward_hook(self, model, input, output):
        self.feature_map.append(output)

    def backward_hook(self, model, grad_input, grad_output):
        self.weight.append(grad_output[0])

    def __call__(self, models, layers, inputs, labels, clip_min=-1, clip_max=1):
        self.weight.clear()
        self.feature_map.clear()
        for model, layer in zip(models, layers):
            if hasattr(model, layer):
                getattr(model, layer).register_backward_hook(self.backward_hook)
                getattr(model, layer).register_forward_hook(self.forward_hook)
            else:
                raise Exception(f'can not find {layer} in {model.__class__.__name__}')

        a = self.a
        g = torch.zeros_like(inputs)
        # transN = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        weights = []
        images = inputs.clone().detach()
        labels = labels.clone().detach()
        adv = images.clone().detach()

        for i in range(self.steps):
            adv.requires_grad = True
            if i == 0:
                for model in models:
                    temp_weight = 0
                    self.weight.clear()
                    for l in range(self.ens):
                        self.feature_map.clear()
                        x_base = torch.zeros_like(inputs)
                        x_base = transN(x_base)
                        temp_noise = np.random.normal(size=inputs.shape, loc=0.0, scale=0.2)
                        temp_noise = torch.from_numpy(temp_noise).to(self.device, dtype=torch.float32)
                        image_tmp = torch.clip(inputs.clone() + temp_noise, -1, 1)
                        image_tmp = inputs.clone()
                        image_tmp = (image_tmp * (1 - l / self.ens) + (l / self.ens) * x_base)
                        logits = model(image_tmp)
                        logits = nn.functional.softmax(logits, 1)
                        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float()
                        score = logits * labels_onehot
                        loss = torch.sum(torch.sum(score, 1))
                        loss.backward()
                    for w in self.weight:
                        temp_weight += w
                    weights.append(temp_weight)
                for idx, weight_tensor in enumerate(weights):
                    weight_tensor = weight_tensor.to(self.device)
                    square = torch.sum(torch.square(weight_tensor), [1, 2, 3], keepdim=True)
                    weight_tensor = -weight_tensor / torch.sqrt(square)
                    weights[idx] = weight_tensor

            base_line = torch.zeros_like(inputs)
            base_line = transN(base_line)
            self.feature_map.clear()
            base_feature = []
            for model in models:
                model(base_line)
            for fm in self.feature_map:
                base_feature.append(fm)
            self.feature_map.clear()

            loss = self.get_NAA_loss(adv, models, weights, base_feature)

            loss.backward()

            adv_grad = adv.grad.clone()
            adv.grad.data.zero_()
            g = self.u * g + (adv_grad / (torch.mean(torch.abs(adv_grad), [1, 2, 3], keepdim=True)))
            adv = adv + a * torch.sign(g)
            if clip_min == None or clip_max == None:
                adv_ = Re_transforms(adv, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                adv = torch.clip(adv_, 0, 1)
                adv = transN(adv)
            else:
                adv = torch.clip(adv, clip_min, clip_max)
            diff = adv - inputs
            noise = torch.clip(diff, -self.esp, self.esp)
            model.zero_grad()
            fin_adv = noise.clone() + inputs
        return fin_adv, noise
