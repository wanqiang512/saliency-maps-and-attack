"""Implementation of sample attack."""
import os
import random
import pretrainedmodels
import torch
from PIL.Image import Image
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
from tqdm import tqdm
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='', help='Input directory with images.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--image_resize", type=int, default=330, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--amplification", type=float, default=10.0, help="To amplifythe step size.")
parser.add_argument("--prob", type=float, default=0.7, help="probability of using diverse inputs.")
opt = parser.parse_args()

transforms = T.Compose([T.CenterCrop(opt.image_width), T.ToTensor()])


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern).cuda()
    return stack_kern, kern_size // 2


def project_noise(x, stack_kern, padding_size):
    # x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = F.conv2d(x, stack_kern, padding=(padding_size, padding_size), groups=3)
    return x


stack_kern, padding_size = project_kern(3)


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


def graph(model, x, gt, x_min, x_max):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    alpha_beta = alpha * opt.amplification
    gamma = alpha_beta
    x.requires_grad = True
    momentum = torch.zeros_like(x).detach().cuda()
    decay = 1.0
    amplification = 0.0
    for i in range(num_iter):
        logits = model(x)
        loss = F.cross_entropy(logits, gt)
        model.zero_grad()
        loss.backward()
        noise = x.grad.data
        # MI-FGSM
        noise = noise / torch.mean(torch.abs(noise), dim=(1, 2, 3), keepdim=True)
        noise = momentum * decay + noise
        momentum = noise

        amplification += alpha_beta * torch.sign(noise)
        cut_noise = torch.clamp(abs(amplification) - eps, 0, 10000.0) * torch.sign(amplification)
        projection = gamma * torch.sign(project_noise(cut_noise, stack_kern, padding_size))
        amplification += projection
        x = x + alpha_beta * torch.sign(noise) + projection
        x = clip_by_tensor(x, x_min, x_max)
        x = x.detach().requires_grad_(True)

    return x.detach()


def save_image(images, names, output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)


def seed_torch(seed):
    """Set a random seed to ensure that the results are reproducible"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def main():
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    model = pretrainedmodels.inception_v3(pretrained=True).eval().cuda()
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    for images, _, gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)
        adv_img = graph(model, images, gt, images_min, images_max)
        adv_img_np = adv_img.detach().cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, _, opt.output_dir)


if __name__ == '__main__':
    seed_torch(1234)
    main()
