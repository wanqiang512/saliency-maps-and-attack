import argparse
import numpy as np
import torch.nn.functional as F
import torch
parser = argparse.ArgumentParser(description='Attack parameters')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the attack')
parser.add_argument('--max_epsilon', type=float, default=16.0, help='Maximum epsilon for the attack')
parser.add_argument('--num_iter', type=int, default=10, help='Number of iterations for the attack')
parser.add_argument('--momentum', type=float, default=1.0, help='Momentum for the attack')
parser.add_argument('--number', type=int, default=20, help='Number of images for the attack')
parser.add_argument('--beta', type=float, default=3.5, help='Beta for the attack')
parser.add_argument('--image_width', type=int, default=299, help='Width of input images')
parser.add_argument('--image_height', type=int, default=299, help='Height of input images')
parser.add_argument('--prob', type=float, default=0.5, help='Probability for applying image transformations')
parser.add_argument('--image_resize', type=int, default=331, help='Size of resized images')
parser.add_argument('--checkpoint_path', type=str, default='./models', help='Path to checkpoint folder')
parser.add_argument('--input_dir', type=str, default='./dev_data/val_rs', help='Path to input directory')
parser.add_argument('--output_dir', type=str, default='./outputs/gra_v3', help='Path to output directory')
FLAGS = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batch_grad(model, x, gt, alpha):
    y = np.random.uniform(-alpha, alpha, size=x.shape)
    x_neighbor = x + torch.from_numpy(y).to(device, dtype=torch.float32)
    x_neighbor.requires_grad = True
    logits = model(x_neighbor)
    loss = F.cross_entropy(logits, gt)
    grad = torch.autograd.grad(loss, x_neighbor, retain_graph=False, create_graph=False)[0]
    return grad


def graph(model, inputs, gt):
    grad = torch.zeros_like(inputs)
    samgrad = torch.zeros_like(inputs)
    m = torch.ones_like(inputs) * 10 / 9.4

    eps = FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    u = FLAGS.momentum
    adv = inputs.clone().detach().to(device)
    for l in range(num_iter):
        adv.requires_grad = True
        logits = model(adv)
        loss = F.cross_entropy(logits, gt)
        adv_grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        adv_neighbor_grad = torch.zeros_like(inputs).to(device)
        for x in range(FLAGS.number):
            adv_neighbor_grad += batch_grad(model, adv, gt, eps * FLAGS.beta) / FLAGS.number

        # Neighbor weighted Correction
        cossim = torch.mean(adv_grad * adv_neighbor_grad, dim=[1, 2, 3]) / \
                 torch.sqrt(torch.mean(adv_grad ** 2, dim=[1, 2, 3])) * torch.sqrt(
            torch.mean(adv_neighbor_grad ** 2, dim=[1, 2, 3]))
        cossim = cossim.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        WG = cossim * adv_grad + (1 - cossim) * adv_neighbor_grad
        noiselast = grad
        noise = u * grad + (WG) / torch.mean(torch.abs(WG), [1, 2, 3], keepdim=True)
        eqm = torch.equal(torch.sign(noiselast), torch.sign(noise)).to(dtype=torch.float32)
        dim = torch.ones(adv.shape) - eqm
        m = m * (eqm + dim * 0.94)
        adv = adv.detach() + alpha * m * noise.sign()
        delta = torch.clip(adv - inputs, -eps, eps)
        adv = torch.clip(inputs + delta, 0, 1).detach()
    return adv
