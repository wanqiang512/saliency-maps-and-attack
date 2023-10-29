import argparse
import torch
from torchvision.transforms import Normalize
import torch.nn.functional as F
from attack_methods import gkern, DI
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='How many images process at one time.')
parser.add_argument('--max_epsilon', type=float, default=16.0, help='max epsilon.')
parser.add_argument('--num_iter', type=int, default=10, help='max iteration.')
parser.add_argument('--u', type=float, default=1.0, help='momentum about the model.')
parser.add_argument('--portion', type=float, default=0.2, help='protion for the mixed image')
parser.add_argument('--size', type=int, default=3, help='Number of randomly sampled images')
parser.add_argument('--image_width', type=int, default=299, help='Width of each input images.')
parser.add_argument('--image_height', type=int, default=299, help='Height of each input images.')
parser.add_argument('--prob', type=float, default=0.5, help='probability of using diverse inputs.')
parser.add_argument('--checkpoint_path', type=str, default='./models', help='Path to checkpoint for pretained models.')
parser.add_argument('--input_dir', type=str, default='./dev_data/val_rs', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory with images.')
FLAGS = parser.parse_args()
T_kernel = gkern(7, 3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def admix(x):
    indices = torch.arange(start=0, end=x.shape[0], dtype=torch.int32)
    return torch.cat([(x + FLAGS.portion * x[torch.randperm(indices.shape[0])]) for _ in range(FLAGS.size)], dim=0)


def TNormalize(x, IsRe=False, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    if not IsRe:
        x = Normalize(mean=mean, std=std)(x)
    elif IsRe:
        # tensor.shape:(3,w.h)
        for idx, i in enumerate(std):
            x[:, idx, :, :] *= i
        for index, j in enumerate(mean):
            x[:, index, :, :] += j
    return x


def graph(model, inputs, gt):
    eps = FLAGS.max_epsilon / 255
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    u = FLAGS.u
    g = torch.zeros_like(inputs).to(device)
    adv = inputs.clone().detach().to(device)
    for l in range(num_iter):
        adv.requires_grad = True
        x_admix = admix(adv)
        x_batch = torch.cat([x_admix, x_admix / 2, x_admix / 4, x_admix / 8, x_admix / 16], dim=0)
        # DI logits = model(DI(x_batch))
        logits = model(x_batch)
        one_hot = torch.cat([F.one_hot(gt, len(logits[0]))] * 5 * FLAGS.size, dim=0)
        loss = F.cross_entropy(logits, one_hot.argmax(dim=1))
        grad = (
            torch.mean(torch.stack(
                torch.chunk(torch.autograd.grad(loss, x_batch, retain_graph=False, create_graph=False)[0], 5), 0)
                       * torch.tensor([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None, None, None, None].cuda(), dim=0)
        )
        grad = torch.sum(torch.stack(torch.chunk(grad, FLAGS.size), 0), dim=0)
        # TI
        grad = F.conv2d(grad, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)
        # MI
        g = u * g + grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        adv = adv + alpha * torch.sign(g)
        delta = torch.clip(adv - inputs, -eps, eps)
        adv = torch.clip(inputs + delta, 0, 1).detach()
    return adv
