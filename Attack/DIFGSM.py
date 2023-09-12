import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize


class DIFGSM:
    r"""
    DI2-FGSM in the paper 'Improving Transferability of Adversarial Examples with Input Diversity'
    u: 衰减因子 u == 0 M-DI-FGSM - > DIFGSM
    diversity_prob: 转换概率  p ==0 M-DI-FGSM -> MIFGSM
    """

    def __init__(
            self,
            model,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            eps=8 / 255,
            alpha=2 / 255,
            steps=10,
            u=0.0,
            resize_rate=0.9,
            diversity_prob=0.5,
            random_start=False
    ):
        self.eps = eps
        self.steps = steps
        self.u = u
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
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

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = images.clone().detach()
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(self.TNormalize(adv_images))
            cost = loss(outputs, labels)
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clip(images + delta, 0, 1).detach_()

        return adv_images
