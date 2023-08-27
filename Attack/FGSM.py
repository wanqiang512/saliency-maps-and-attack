import torch
import torch.nn as nn

class FGSM:
    r"""
    Distance Measure : Linf
    """
    def __init__(self, model, device=None, eps=8 / 255):
        self.eps = eps
        self.model = model
        self.device = device

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        images.requires_grad = True
        outputs = self.model(images)
        cost = loss(outputs, labels)
        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
