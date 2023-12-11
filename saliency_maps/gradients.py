from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.autograd import Variable


class GradCam:
    def __init__(self, model):
        super(GradCam, self).__init__()
        self.model = model
        self.model.eval()  # model have to get .eval() for evaluation.

    def normalization(self, x):
        x -= x.min()
        if x.max() <= 0.:
            x /= 1.  # to avoid Nan
        else:
            x /= x.max()
        return x

    def get_names(self):
        """ function to get names of layers in the model. """
        for name, module in self.model.named_modules():
            print(name, '//', module)

    def forward_hook(self, name, input_hook=False):
        def save_forward_hook(module, input, output):
            if input_hook:
                self.forward_out[name] = input[0].detach()
            else:
                self.forward_out[name] = output.detach()

        return save_forward_hook

    def backward_hook(self, name, input_hook=False):
        def save_backward_hook(module, grad_input, grad_output):
            if input_hook:
                self.backward_out[name] = grad_input[0].detach()
            else:
                self.backward_out[name] = grad_output[0].detach()

        return save_backward_hook

    def get_gradient(self, input_TensorImage, target_layers, target_label=None, counter=False, input_hook=False):
        """
        Get backward-propagation gradient.

        :param input_TensorImage (tensor): Input Tensor image with [1, c, h, w].
        :param target_layers (str, list): Names of target layers. Can be set to string for a layer, to list for multiple layers, or to "All" for all layers in the model.
        :param target_label (int, tensor): Target label. If None, will determine index of highest label of the model's output as the target_label.
                                            Can be set to int as index of output, or to a Tensor that has same shape with output of the model. Default: None
        :param counter (bool): If True, will get negative gradients only for conterfactual explanations. Default: True
        :param input_hook (bool): If True, will get input features and gradients of target layers instead of output. Default: False
        :return (list): A list including gradients of Gradcam for target layers
        """
        if not isinstance(input_TensorImage, torch.Tensor):
            raise NotImplementedError('input_TensorImage is a must torch.Tensor format with [..., C, H, W]')
        self.model.zero_grad()
        self.forward_out = {}
        self.backward_out = {}
        self.handlers = []
        self.gradients = []
        self.target_layers = target_layers

        if not input_TensorImage.size()[0] == 1: raise NotImplementedError("batch size of input_TensorImage must be 1.")
        if not target_layers == 'All':
            if isinstance(target_layers, str) or not isinstance(target_layers, Iterable):
                self.target_layers = [self.target_layers]
                for target_layer in self.target_layers:
                    if not isinstance(target_layer, str):
                        raise NotImplementedError(
                            " 'Target layers' or 'contents in target layers list' are must string format.")

        for name, module in self.model.named_modules():
            if target_layers == 'All':
                if isinstance(module, nn.Conv2d):
                    self.handlers.append(module.register_forward_hook(self.forward_hook(name, input_hook)))
                    self.handlers.append(module.register_backward_hook(self.backward_hook(name, input_hook)))
            else:
                if name in self.target_layers:
                    self.handlers.append(module.register_forward_hook(self.forward_hook(name, input_hook)))
                    self.handlers.append(module.register_backward_hook(self.backward_hook(name, input_hook)))

        output = self.model(input_TensorImage)

        if target_label is None:
            target_tensor = torch.zeros_like(output)
            target_tensor[0][int(torch.argmax(output))] = 1.
        else:
            if isinstance(target_label, int):
                target_tensor = torch.zeros_like(output)
                target_tensor[0][target_label] = 1.
            elif isinstance(target_label, torch.Tensor):
                # if not target_label.dim() == output.dim():
                #     raise NotImplementedError('Dimension of output and target label are different')
                target_tensor = torch.zeros_like(output)
                target_tensor[0][target_label.item()] = 1.
        output.backward(target_tensor)

        self.model.zero_grad()
        for handle in self.handlers:
            handle.remove()

        def process():
            grads = self.backward_out[name]
            if counter:
                grads = torch.clamp(grads, max=0.)
                grads *= -1.
            weight = torch._adaptive_avg_pool2d(grads, 1)
            gradient = self.forward_out[name] * weight
            gradient = gradient.sum(dim=1, keepdim=True)
            gradient = F.relu(gradient)
            gradient = self.normalization(gradient)
            self.gradients.append(gradient)

        if not target_layers == 'All':
            for name in self.target_layers:
                process()
        else:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    process()

        return self.gradients


class GradCamplusplus:
    def __init__(self, model):
        super(GradCamplusplus, self).__init__()
        self.model = model
        self.model.eval()  # model have to get .eval() for evaluation.

    def normalization(self, x):
        x -= x.min()
        if x.max() <= 0.:
            x /= 1.  # to avoid Nan
        else:
            x /= x.max()
        return x

    def get_names(self):
        """ function to get names of layers in the model. """
        for name, module in self.model.named_modules():
            print(name, '//', module)

    def forward_hook(self, name, input_hook=False):
        def save_forward_hook(module, input, output):
            if input_hook:
                self.forward_out[name] = input[0].detach()
            else:
                self.forward_out[name] = output.detach()

        return save_forward_hook

    def backward_hook(self, name, input_hook=False):
        def save_backward_hook(module, grad_input, grad_output):
            if input_hook:
                self.backward_out[name] = grad_input[0].detach()
            else:
                self.backward_out[name] = grad_output[0].detach()

        return save_backward_hook

    def get_gradient(self, input_TensorImage, target_layers, target_label=None, counter=False, input_hook=False):
        """
        Get backward-propagation gradient.

        :param input_TensorImage (tensor): Input Tensor image with [1, c, h, w].
        :param target_layers (str, list): Names of target layers. Can be set to string for a layer, to list for multiple layers, or to "All" for all layers in the model.
        :param target_label (int, tensor): Target label. If None, will determine index of highest label of the model's output as the target_label.
                                            Can be set to int as index of output, or to a Tensor that has same shape with output of the model. Default: None
        :param counter (bool): If True, will get negative gradients only for conterfactual explanations. Default: True
        :param input_hook (bool): If True, will get input features and gradients of target layers instead of output. Default: False
        :return (list): A list including gradients of Gradcam for target layers
        """
        if not isinstance(input_TensorImage, torch.Tensor):
            raise NotImplementedError('input_TensorImage is a must torch.Tensor format with [..., C, H, W]')
        self.model.zero_grad()
        self.forward_out = {}
        self.backward_out = {}
        self.handlers = []
        self.gradients = []
        self.target_layers = target_layers

        if not input_TensorImage.size()[0] == 1: raise NotImplementedError("batch size of input_TensorImage must be 1.")
        if not target_layers == 'All':
            if isinstance(target_layers, str) or not isinstance(target_layers, Iterable):
                self.target_layers = [self.target_layers]
                for target_layer in self.target_layers:
                    if not isinstance(target_layer, str):
                        raise NotImplementedError(
                            " 'Target layers' or 'contents in target layers list' are must string format.")

        for name, module in self.model.named_modules():
            if target_layers == 'All':
                if isinstance(module, nn.Conv2d):
                    self.handlers.append(module.register_forward_hook(self.forward_hook(name, input_hook)))
                    self.handlers.append(module.register_backward_hook(self.backward_hook(name, input_hook)))
            else:
                if name in self.target_layers:
                    self.handlers.append(module.register_forward_hook(self.forward_hook(name, input_hook)))
                    self.handlers.append(module.register_backward_hook(self.backward_hook(name, input_hook)))

        output = self.model(input_TensorImage)

        if target_label is None:
            target_tensor = torch.zeros_like(output)
            target_tensor[0][int(torch.argmax(output))] = 1.
        else:
            if isinstance(target_label, int):
                target_tensor = torch.zeros_like(output)
                target_tensor[0][target_label] = 1.
            elif isinstance(target_label, torch.Tensor):
                # if not target_label.dim() == output.dim():
                #     raise NotImplementedError('Dimension of output and target label are different')
                target_tensor = torch.zeros_like(output)
                target_tensor[0][target_label.item()] = 1.
        output.backward(target_tensor)

        self.model.zero_grad()
        for handle in self.handlers:
            handle.remove()

        def process():
            features = self.forward_out[name]
            grads = self.backward_out[name]
            if counter:
                grads *= -1.
            relu_grads = F.relu(grads)
            alpha_numer = grads.pow(2)
            alpha_denom = 2. * grads.pow(2) + grads.pow(3) * features.sum(dim=-1, keepdim=True).sum(dim=-2,
                                                                                                    keepdim=True)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
            alpha = alpha_numer / alpha_denom
            weight = (alpha * relu_grads).sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
            gradient = features * weight
            gradient = gradient.sum(dim=1, keepdim=True)
            gradient = F.relu(gradient)
            gradient = self.normalization(gradient)
            self.gradients.append(gradient)

        if not target_layers == 'All':
            for name in self.target_layers:
                process()
        else:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    process()

        return self.gradients


class SmoothGrad:
    """
    SmoothGrad
    example:
     >>> SG = saliency_maps.SmoothGrad(model, 20, False, 0.2, device)
     >>> grad = SG(images, labels)
    """

    def __init__(
            self,
            model,
            ens,
            magnitude=False,
            scale=0.2,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        self.device = device
        self.model = model.eval().to(device)
        self.ens = ens
        self.scale = scale
        self.magnitude = magnitude

    def __call__(self, x, idx=None, *args, **kwargs):
        score = self.model(x)
        if idx is None:
            prob, idx = torch.max(score, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))
        avg_gradients = torch.zeros_like(x)
        for l in range(self.ens):
            noise = np.random.normal(0, self.scale, size=x.shape)
            noise = torch.from_numpy(noise).to(self.device, dtype=torch.float32)
            x_plus_noise = x + noise
            x_plus_noise = Variable(x_plus_noise, requires_grad=True)
            score = self.model(x_plus_noise)
            score = torch.softmax(score, dim=1)
            one_hot = F.one_hot(idx, len(score[0])).float()
            loss = torch.sum(score * one_hot)
            self.model.zero_grad()
            loss.backward()
            if self.magnitude:
                avg_gradients += x_plus_noise.grad * x_plus_noise.grad
            else:
                avg_gradients += x_plus_noise.grad
        avg_gradients = avg_gradients / self.ens
        return avg_gradients


class BlurIG:
    def __init__(
            self,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        self.device = device

    """A CoreSaliency class that implements integrated gradients along blur path.

    https://arxiv.org/abs/2004.03383

    Generates a saliency mask by computing integrated gradients for a given input
    and prediction label using a path that successively blurs the image.
    """

    def gaussian_blur(self, image, sigma):
        """Returns Gaussian blur filtered 3d (WxHxC) image.

        Args:
          image: 3 dimensional ndarray / input image (W x H x C).
          sigma: Standard deviation for Gaussian blur kernel.
        """
        if isinstance(image, torch.Tensor):
            image_numpy = image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
        if sigma == 0:
            return image
        x = ndimage.gaussian_filter(image_numpy, sigma=[sigma, sigma, 0], mode="constant").transpose(2, 0, 1)
        return torch.from_numpy(x).unsqueeze(0).to(self.device, dtype=torch.float32)

    def __call__(
            self,
            model,
            inputs,
            labels,
            max_sigma=50,
            steps=100,
            grad_step=0.01,
            sqrt=False,
            scale=0.2
    ):

        if sqrt:
            sigmas = [math.sqrt(float(i) * max_sigma / float(steps)) for i in range(0, steps + 1)]
        else:
            sigmas = [float(i) * max_sigma / float(steps) for i in range(0, steps + 1)]

        step_vector_diff = [sigmas[i + 1] - sigmas[i] for i in range(0, steps)]

        total_gradients = torch.zeros_like(inputs).to(self.device)

        for i in range(steps):
            noise = np.random.normal(0.0, 0.2, size=inputs.shape)
            temp = inputs.clone() + torch.from_numpy(noise).to(self.device, dtype=torch.float32)
            x_step = self.gaussian_blur(temp, sigmas[i])
            x_step = Variable(x_step, requires_grad=True)
            gaussian_gradient = (self.gaussian_blur(temp, sigmas[i] + grad_step) - x_step) / grad_step
            lotigs = model(x_step)
            lotigs = F.softmax(lotigs, dim=1)
            one_hot = F.one_hot(labels, len(lotigs[0]))
            loss = torch.sum(lotigs * one_hot)
            model.zero_grad()
            loss.backward()
            grad = x_step.grad.data
            tmp = (step_vector_diff[i] * torch.multiply(gaussian_gradient, grad))
            total_gradients += tmp

        total_gradients *= -1.0
        return total_gradients



class NoiseGrad_PlusPlus:
    def __init__(
            self,
            model,
            weights,
            mean: float = 1.0,
            std: float = 0.2,
            sg_mean: float = 0.0,
            sg_std: float = 0.2,
            n: int = 20,
            noise_type: str = "multiplicative",
            verbose: bool = True,
    ):
        """
        Initialize the explanation-enhancing method: NoiseGrad.
        Paper:

        Args:
            model (torch model): a trained model
            weights (dict):
            mean (float): mean of the distribution (often referred to as mu)
            std (float): standard deviation of the distribution (often referred to as sigma)
            n (int): number of Monte Carlo rounds to sample models
            noise_type (str): the type of noise to add to the model parameters, either additive or multiplicative
            verbose (bool): print the progress of explanation enchancement (default True)
        """

        self.std = std
        self.mean = mean
        self.model = model
        self.n = n
        self.weights = weights
        self.noise_type = noise_type
        self.verbose = verbose
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.distribution = torch.distributions.normal.Normal(
            loc=self.mean, scale=self.std
        )
        self.sg_std = sg_std
        self.sg_mean = sg_mean
        print("NoiseGrad++ initialized.")

    def sample(self):
        self.model.load_state_dict(self.weights)
        # If std is not zero, loop over each layer and add Gaussian noise.
        if not self.std == 0.0:
            with torch.no_grad():
                for layer in self.model.parameters():
                    if self.noise_type == "additive":
                        layer.add_(
                            self.distribution.sample(layer.size()).to(layer.device)
                        )
                    elif self.noise_type == "multiplicative":
                        layer.mul_(
                            self.distribution.sample(layer.size()).to(layer.device)
                        )
                    else:
                        print(
                            "Set NoiseGrad attribute 'noise_type' to either 'additive' or 'multiplicative' (str)."
                        )

    def __call__(self, inputs: Optional[torch.tensor], targets: Optional[torch.tensor], **kwargs):
        inputs = inputs.clone().detach().to(self.device)
        targets = targets.clone().detach().to(self.device)
        avg_gradients = torch.zeros_like(inputs).to(self.device)
        for i in (tqdm(range(self.n)) if self.verbose else range(self.n)):  # create a series of models
            self.sample()
            self.model.to(self.device)
            noise = np.random.normal(self.sg_mean, self.sg_std, size=inputs.shape)
            noise = torch.from_numpy(noise).to(self.device, dtype=torch.float32)
            inputs_noise = inputs + noise
            inputs_noise = Variable(inputs_noise, requires_grad=True)
            logits = self.model(inputs_noise)
            score = F.softmax(logits, dim=1)
            one_hot = F.one_hot(targets, len(score[0])).float()
            loss = torch.sum(score * one_hot)
            grad = torch.autograd.grad(loss, inputs_noise, create_graph=False, retain_graph=False)[0]
            avg_gradients += grad
            avg_gradients = avg_gradients / self.n
        return avg_gradients

