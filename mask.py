"""
@author: Britney(wanqiang512)
@software: PyCharm
@file: mask.py
@time: 2023/10/22 0:21
"""
import math
import random
import numpy as np
import torch.nn.functional as F
import torch
from PIL import Image
from torch import Tensor

""""""""""""""""""""""""""""""""""""""""""
# mask 数据增强
""""""""""""""""""""""""""""""""""""""""""


def drop_block2d(
        input: Tensor, block_size: int, p: float = 0.1, training: bool = True,
) -> Tensor:
    """
    Implements DropBlock2d from `"DropBlock: A regularization method for convolutional networks"
    <https://arxiv.org/abs/1810.12890>`.

    Args:
        input (Tensor[N, C, H, W]): The input tensor or 4-dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): Probability of an element to be dropped.
        block_size (int): Size of the block to drop.
        inplace (bool): If set to ``True``, will do this operation in-place. Default: ``False``.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-6.
        training (bool): apply dropblock if is ``True``. Default: ``True``.

    Returns:
        Tensor[N, C, H, W]: The randomly zeroed tensor after dropblock.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}.")
    if input.ndim != 4:
        raise ValueError(f"input should be 4 dimensional. Got {input.ndim} dimensions.")
    if not training or p == 0.0:
        return input
    if block_size != 0:
        N, C, H, W = input.size()
        block_size = min(block_size, W, H)
        # compute the gamma of Bernoulli distribution
        gamma = (p * H * W) / ((block_size ** 2) * ((H - block_size + 1) * (W - block_size + 1)))
        noise = torch.empty((N, C, H - block_size + 1, W - block_size + 1), dtype=input.dtype, device=input.device)
        noise.bernoulli_(gamma)

        noise = F.pad(noise, [block_size // 2] * 4, value=0)
        noise = F.max_pool2d(noise, stride=(1, 1), kernel_size=(block_size, block_size), padding=block_size // 2)
        noise = 1 - noise
        input = input * noise

    return input


class Grid(object):
    def __init__(self, d1, d2, rotate, ratio, mode, prob):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))

        d = np.random.randint(self.d1, self.d2)
        # d = self.d

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]
        if self.mode == 1:
            mask = 1 - mask
        mask = torch.from_numpy(mask).float().cuda()
        mask = mask.expand_as(img)
        img = img * mask

        return img


class GridMask:
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=1, prob=1):
        super(GridMask, self).__init__()
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)

    def __call__(self, x):
        n, c, h, w = x.size()
        y = []
        for i in range(n):
            y.append(self.grid(x[i]))
        y = torch.cat(y).view(n, c, h, w)
        return y
