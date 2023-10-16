"""Implementation of sample attack."""
import collections
import os
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels

T_kernel = gkern(7, 3)

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./adv_img/', help='Source Models.')
parser.add_argument('--model_name', type=str, default='inceptionv3', help='Output directory with adversarial images.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=20, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--N", type=int, default=10, help="The number of Spectrum Transformations")
parser.add_argument("--rho", type=float, default=0.5, help="Tuning factor")
parser.add_argument("--sigma", type=float, default=16.0, help="Std of random noise")
parser.add_argument('--layer', type=str, default='AuxLogits', help='layer of Source Models.')
opt = parser.parse_args()
torch.backends.cudnn.benchmark = True

transforms = T.Compose([T.CenterCrop(opt.image_width), T.ToTensor()])


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def admix(x):
    indices = torch.arange(start=0, end=x.shape[0], dtype=torch.int32)
    return torch.cat([(x + 0.2 * x[torch.randperm(indices.shape[0])]) for _ in range(3)], dim=0)


def graph(model, x, gt):
    eps = opt.max_epsilon / 255.0
    num_iter = 10
    alpha = eps / num_iter

    decay = 1.0
    adv_images = x.clone().detach()
    momentum = torch.zeros_like(x).detach().cuda()
    y = torch.cat([gt] * 5 * 3, dim=0)
    for i in range(num_iter):
        adv_images.requires_grad = True
        x_admix = admix(adv_images)
        x_batch = torch.cat([x_admix, x_admix / 2., x_admix / 4., x_admix / 8., x_admix / 16.], dim=0)

        output = svd_inv3(model[1], model[0](DI(x_batch)))
        loss = F.cross_entropy(output, y)

        grad = torch.mean(
            torch.stack(torch.chunk(torch.autograd.grad(loss, x_batch, retain_graph=False, create_graph=False)[0], 5),
                        0)
            * torch.tensor([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None, None, None, None].cuda(), dim=0)

        grad = torch.sum(torch.stack(torch.chunk(grad, 3), 0), dim=0)

        # # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        grad = F.conv2d(grad, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        # # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + momentum * decay
        momentum = grad

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - x, min=-eps, max=eps)
        adv_images = torch.clamp(x + delta, min=0, max=1).detach()

    return adv_images


def save_image(images, names, output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + '/' + name)


def main():
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    model = torch.nn.Sequential(Normalize(mean, std),
                                pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())

    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    for images, images_ID, gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        adv_img = graph(model, images, gt)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)


if __name__ == '__main__':
    main()