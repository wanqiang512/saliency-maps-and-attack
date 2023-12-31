import torch
import torch.nn as nn
from torchvision.transforms import Normalize


class mifgsm:
    def __init__(
            self,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            eps=16 / 255,
            alpha=1.6 / 255,
            steps=10,
            u=1.0
    ):
        self.eps = eps
        self.steps = steps
        self.u = u  # u
        self.alpha = alpha
        self.device = device

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

    def __call__(self, model, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = images.clone().detach()
        momentum = torch.zeros_like(images).detach().to(self.device)  # g
        loss = nn.CrossEntropyLoss()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = model(self.TNormalize(adv_images))
            cost = loss(outputs, labels)
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad),
                                     dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.u
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clip(images + delta, 0, 1).detach_()

        return adv_images
