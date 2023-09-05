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

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:  # BIM 和 PGD 区别
            # Starting at a uniformly random point
            adv_images = adv_images + \
                         torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(self.TNormalize(adv_images))
            cost = loss(outputs, labels)
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clip(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clip(images + delta, 0, 1).detach_()

        return adv_images
