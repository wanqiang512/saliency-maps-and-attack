import torch
import torch.nn as nn
from torchvision.transforms import Normalize


class IFGSM:
    def __init__(self,
                 steps,
                 eps,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.steps = steps
        self.device = device
        self.eps = eps

    def TNormalize(self, x, IsRe, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        if not IsRe:
            x = Normalize(mean=mean, std=std)(x)
        elif IsRe:
            # tensor.shape:(3,w.h)
            for idx, i in enumerate(std):
                x[:, idx, :, :] *= i
            for index, j in enumerate(mean):
                x[:, index, :, :] += j
        return x

    def attack(self, model, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv = images.clone().detach()
        alpha = self.eps / self.steps

        for i in range(self.steps):
            adv.requires_grad = True
            logits = model(self.TNormalize(adv))
            ce_loss = nn.CrossEntropyLoss()  # loss = F.nll_loss(logits, labels)
            loss = ce_loss(logits, labels)
            loss.backward()
            adv = adv + alpha * torch.sign(adv.grad)
            model.zero_grad()
            diff = adv - images
            delta = torch.clamp(diff, -self.eps, self.eps)
            adv = torch.clip(delta + images, 0, 1).detach_()
        return adv
