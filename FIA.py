import torch
import torch.nn as nn
import numpy as np

__all__ = ['FIA']

from torch import tensor


class FIA:
    def __init__(self, eps=8 / 255, steps=10, u=1, a=1.6 / 255, ens=30, drop_pb=0.7,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.eps = eps
        self.device = device
        self.steps = steps
        self.u = u
        self.a = a
        self.ens = ens
        self.drop_pb = drop_pb
        self.feature_map = {}
        self.weight = {}

    def get_FIA_loss(self, x, model, weight, layer):
        self.feature_map.clear()
        loss = 0
        logits = model(x)
        attribution = self.feature_map[layer] * weight
        loss = torch.sum(attribution) / attribution.numel()
        return loss

    def save_fmaps(self, key: str) -> None:
        def forward_hook(model: nn.Module, input: tensor, output: tensor) -> None:
            self.feature_map[key] = output

        return forward_hook

    def save_weight(self, key: str) -> None:
        def backward_hook(model: nn.Module, grad_input: tensor, grad_output: tensor) -> None:
            self.weight[key] = grad_output[0]

        return backward_hook

    def __call__(self, model: nn.Module, inputs: torch.tensor, labels: torch.tensor, layer: str):
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
        adv = images.clone().detach().to(self.device)
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
                    logits = model(image_tmp)
                    logits = nn.functional.softmax(logits, 1)
                    labels_onehot = torch.nn.functional.one_hot(labels, len(logits[0])).float()
                    score = logits * labels_onehot
                    loss = torch.sum(score)
                    loss.backward()
                    temp_weight += self.weight[layer].clone().detach()

                temp_weight.to(self.device)
                square = torch.sum(torch.square(temp_weight), [1, 2, 3], keepdim=True)
                weight = - temp_weight / torch.sqrt(square)

            loss = self.get_FIA_loss(adv, model, weight, layer)
            loss.backward()

            adv_grad = adv.grad.clone()
            adv.grad.data.zero_()
            g = self.u * g + (adv_grad / (torch.mean(torch.abs(adv_grad), [1, 2, 3], keepdim=True)))
            adv = adv.detach_() + a * torch.sign(g)
            diff = adv - inputs
            delta = torch.clip(diff, -self.eps, self.eps)
            adv = torch.clip(inputs + delta, 0, 1)

        return adv
