from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import Normalize


class GradCam:
    def __init__(self, model):
        super(GradCam, self).__init__()
        self.model = model
        self.model.eval()  # model have to get .eval() for evaluation.

    def TNormalize(self, x, IsRe=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        if not IsRe:
            x = Normalize(mean=mean, std=std)(x)
        elif IsRe:
            # tensor.shape:(3,w.h)
            for idx, i in enumerate(std):
                x[:, idx, :, :] *= i
            for index, j in enumerate(mean):
                x[:, index, :, :] += j
        return x

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

        output = self.model(self.TNormalize(input_TensorImage))

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
