import torch
import torch.nn as nn
from torchvision.transforms import Normalize


class PGD(object):
    r"""
    Distance Measure : Linf
    """

    def __init__(
            self,
            model,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            eps=8 / 255,
            alpha=2 / 255,
            steps=10,
            random_start=True
    ):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.model = model
        self.device = device

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

    def forward(self, images, labels, clip_min=None, clip_max=None):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:  # BIM 和 PGD 区别
            # Starting at a uniformly random point
            adv_images = adv_images + \
                         torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            # adv_images = self.TNormalize(adv_images, IsRe=True)
            adv_images = torch.clamp(adv_images, min=clip_min, max=clip_max).detach()
            # adv_images = self.TNormalize(adv_images)

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            cost = loss(outputs, labels)
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = (images + delta).detach_()
            if clip_min is None and clip_max is None:
                adv_images = self.TNormalize(adv_images, IsRe=True)
                adv_images.clip(0, 1)
                adv_images = self.TNormalize(adv_images, IsRe=False)
            else:
                adv_images.clip(clip_min, clip_max)

        return adv_images
