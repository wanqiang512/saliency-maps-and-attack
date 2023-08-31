import torch
import numpy as np
import torch.nn as nn
from torch import tensor
from torchvision import transforms
import torchattacks as ta
from torchvision.transforms import Normalize


def image_transform(x):
    return x


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
    stack_kern = np.expand_dims(stack_kern, 3)
    return stack_kern, kern_size // 2


def normalize(grad, opt=2):
    if opt == 0:
        nor_grad = grad
    elif opt == 1:
        abs_sum = torch.sum(torch.abs(grad), [1, 2, 3], keepdim=True)
        nor_grad = grad / abs_sum
    elif opt == 2:
        square = torch.sum(torch.square(grad), [1, 2, 3], keepdim=True)
        nor_grad = grad / torch.sqrt(square)
    return nor_grad


class NAA(nn.Module):
    def __init__(self,
                 net,
                 use_cuda: True,
                 num_iter=10,
                 alpha=1.6 / 255,
                 max_epsilon=16 / 255,
                 momentum=1,
                 virtual_image_num=30, ):
        """
        Args:
            net: model
            num_iter: Number of iterations.
            max_epsilon: Maximum size of adversarial perturbation
            momentum: Momentum
            virtual_image_num: Number of aggregated n.
        """
        super().__init__()
        self.useDIM = False
        self.usePIM = False
        self.device = 'cuda:0' if torch.cuda.is_available() and use_cuda else 'cpu'
        self.net = net.to(self.device)
        self.num_iter = num_iter
        self.alpha = alpha
        self.max_epsilon = max_epsilon
        self.momentum = momentum
        self.virtual_image_num = virtual_image_num
        self.gamma = 1.0

    def NAALoss(self, adv_feature, base_feature, weights):
        gamma = self.gamma
        attribution = (adv_feature - base_feature) * weights
        blank = torch.zeros_like(attribution)
        positive = torch.where(attribution >= 0, attribution, blank)
        negative = torch.where(attribution < 0, attribution, blank)
        # Transformation: Linear transformation performs the best
        balance_attribution = positive + gamma * negative
        loss = torch.sum(balance_attribution) / (base_feature.shape[0] * base_feature.shape[1])

        return loss

    def project_noise(self, x, stack_kern, kern_size):
        import torch.nn.functional as F
        padding = (kern_size, kern_size)
        x = F.pad(x, padding, "constant")
        out_channels = stack_kern.shape[0]
        kernel_size = (kern_size, kern_size)
        x = F.conv2d(x, stack_kern, stride=1, padding=0, groups=out_channels)
        return x

    def TNormalize(self, x, IsRe, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        if not IsRe:
            x = Normalize(mean=mean, std=std)
        elif IsRe:
            # tensor.shape:(3,w.h)
            for idx, i in enumerate(std):
                x[:, idx, :, :] *= i
            for index, j in enumerate(mean):
                x[:, index, :, :] += j
        return x

    def set_DIM_ens(self,
                    image_size=224,
                    image_resize=331,
                    prob=0.7):
        """
        Args:
            image_size: size of each input images
            image_resize: size of each diverse images
            prob: Probability of using diverse inputs.
        """
        self.useDIM = True
        self.image_size = image_size,
        self.image_resize = image_resize,
        self.prob = prob

    def set_PIM_ens(self,
                    Pkern_size=3,
                    gamma=0.5,
                    amplification_factor=2.5):
        """
        Args:
            Pkern_size: To amplifythe step size.
            gamma: The gamma parameter
            amplification_factor: Kernel size of PIM
        """
        self.P_kern, self.kern_size = project_kern(Pkern_size)
        # self.gamma = gamma,
        self.amplification_factor = amplification_factor

    def forward(self, x, y, clip_min=None, clip_max=None):
        images = x.clone().detach().to(self.device)
        labels = y.clone().detach().to(self.device)
        adv_images = x.clone().detach().to(self.device)

        grad_np = torch.zeros_like(images)
        amplification_np = torch.zeros_like(images)
        weight_np = 0

        for iter in range(self.num_iter):

            adv_images.requires_grad = True

            if iter == 0:
                if self.virtual_image_num == 0:
                    adv_images = image_transform(adv_images)
                    feamap, _, logits = self.net(adv_images)
                    weight_np += torch.autograd.grad(torch.nn.functional.softmax(logits) * labels, feamap)[0]

                for l in range(int(self.virtual_image_num)):
                    # x_base = np.array([0.0, 0.0, 0.0])
                    x_base = torch.zeros_like(images).to(self.device)
                    x_base = image_transform(x_base)
                    images_base = image_transform(images)
                    images_base += (torch.randn_like(images) * 0.2 + 0)
                    images_base = images_base * (1 - l / self.virtual_image_num) + (l / self.virtual_image_num) * x_base
                    feamap, _, logits = self.net(images_base)
                    one_hot = torch.nn.functional.one_hot(labels, num_classes=len(logits[0])).to(self.device)
                    logits = torch.softmax(logits, dim=1)
                    weight_np += torch.autograd.grad(outputs=logits, inputs=feamap, grad_outputs=one_hot)[0]

                weight_np = -normalize(weight_np, 2)

            images_base = image_transform(torch.zeros_like(images))
            base_feamap, _, _ = self.net(images_base)

            adv_feamap, _, _ = self.net(adv_images)
            cost = self.NAALoss(adv_feamap, base_feamap, weight_np)
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            grad = grad / torch.mean(torch.abs(grad), [1, 2, 3], keepdim=True)
            grad = self.momentum * grad_np + grad
            grad_np = grad.clone().detach().to(self.device)

            if self.usePIM:
                self.set_PIM_ens()
                # amplification factor
                alpha_beta = self.alpha * self.amplification_factor
                gamma = self.gamma * alpha_beta
                # Project cut noise
                amplification_np += alpha_beta * torch.sign(grad)
                cut_noise = torch.clip(abs(amplification_np) -
                                       self.max_epsilon, 0.0, 10000.0) * torch.sign(amplification_np)
                projection = gamma * torch.sign(self.project_noise(cut_noise, self.P_kern, self.kern_size))
                amplification_np += projection
                adv_input_update = adv_input_update + alpha_beta * torch.sign(grad) + projection
            else:
                adv_images = adv_images + self.alpha * torch.sign(grad)

            # torch.clip(adv_images, images - self.max_epsilon,
            #            images + self.max_epsilon)
            delta = torch.clip(adv_images - images, -self.max_epsilon, self.max_epsilon)
            adv_images = (images + delta).detach_()
            if clip_max is None and clip_min is None:
                adv_images = self.TNormalize(adv_images, True)
                adv_images = adv_images.clip(0, 1)
                adv_images = self.TNormalize(adv_images, False)
            else:
                adv_images = adv_images.clip(clip_min, clip_max)

        return adv_images
