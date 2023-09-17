import numpy as np
import torch
import torch.nn.functional as F

from Normalize import Normalize


class SSA:
    "Spectrum Simulation Attack (ECCV'2022 ORAL)"

    def __init__(
            self,
            eps=16 / 255,
            iters=10,
            ens=20,
            sigma=16 / 255,
            u=1.0,
            rho=0.5
    ):
        self.eps = eps
        self.iters = iters
        self.ens = ens
        self.sigma = sigma
        self.u = u
        self.rho = rho
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = eps / iters

    def dct(self, x, norm=None):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.fft.fft(v)

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
        V = Vc.real * W_r - Vc.imag * W_i
        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)
        return V

    def idct(self, X, norm=None):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

        Our definition of idct is that idct(dct(x)) == x

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
        tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
        v = torch.fft.ifft(tmp)

        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape).real

    def dct_2d(self, x, norm=None):
        """
        2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last 2 dimensions
        """
        X1 = self.dct(x, norm=norm)
        X2 = self.dct(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)

    def idct_2d(self, X, norm=None):
        """
        The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

        Our definition of idct is that idct_2d(dct_2d(x)) == x

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last 2 dimensions
        """
        x1 = self.idct(X, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)

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

    def maps(self):
        """saliency maps in frencymaps"""
        """grad_all = 0
        for images, images_ID, gt_cpu in tqdm(data_loader):
            gt = gt_cpu.cuda()
            images = images.cuda()
            img_dct = dct.dct_2d(images)
            img_dct = V(img_dct, requires_grad=True)
            img_idct = dct.idct_2d(img_dct)

            output_ = model(img_idct)
            loss = F.cross_entropy(output, gt)
            loss.backward()
            grad = img_dct.grad.data
            grad = grad.mean(dim=1).abs().sum(dim=0).cpu().numpy()
            grad_all = grad_all + grad

        x = grad_all / 1000.0
        x = (x - x.min()) / (x.max() - x.min())
        g1 = sns.heatmap(x, cmap="rainbow")
        g1.set(yticklabels=[])  # remove the tick labels
        g1.set(ylabel=None)  # remove the axis label
        g1.set(xticklabels=[])  # remove the tick labels
        g1.set(xlabel=None)  # remove the axis label
        g1.tick_params(left=False)
        g1.tick_params(bottom=False)
        sns.despine(left=True, bottom=True)
        plt.show()
        plt.savefig("fig.png")"""

    def __call__(self, model, inputs, labels, *args, **kwargs):
        adv = inputs.clone().detach().to(self.device)
        inputs_min = self.clip_by_tensor(inputs - self.eps, 0.0, 1.0)
        inputs_max = self.clip_by_tensor(inputs + self.eps, 0.0, 1.0)
        for l in range(self.iters):
            noise = 0
            for x in range(self.ens):
                gauss = torch.randn(inputs.shape) * self.sigma
                gauss = gauss.to(self.device)
                dct = self.dct(adv + gauss).to(self.device)
                mask = (torch.rand_like(adv) * 2 * self.rho + 1 - self.rho).to(self.device)
                idct = self.idct(dct * mask)
                idct.requires_grad = True
                logits = model(idct)
                model.zero_grad()
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                noise += idct.grad.data
            noise = noise / self.ens
            adv = adv + self.alpha * noise.sign()
            adv = self.clip_by_tensor(adv, inputs_min, inputs_max)
        return adv.clone().detach()
