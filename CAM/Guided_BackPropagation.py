import torch
from torch import nn
from torchvision.transforms import Normalize


class Guided_BackPropagation():
    def __init__(self, model):
        super(Guided_BackPropagation, self).__init__()
        self.model = model
        self.model.eval()

    def normalization(self, x):
        x -= x.min()
        if x.max() <= 0.:
            x /= 1.  # to avoid Nan
        else:
            x /= x.max()
        return x

    def relu_backward_hook(self, module, grad_input, grad_output):
        return (torch.clamp(grad_input[0], min=0.),)

    def get_gradient(self, input_TensorImage, target_label=None):
        """
        :param input_TensorImage (tensor): Input Tensor image with [1, c, h, w].
        :param target_label (int, tensor): Target label. If None, will determine index of highest label of the model's output as the target_label.
                                            Can be set to int as index of output, or to a Tensor that has same shape with output of the model. Default: None
        :return (tensor): Guided-BackPropagation gradients of the input image.
        """
        self.model.zero_grad()
        self.guided_gradient = None
        self.handlers = []
        self.gradients = []
        self.input_TensorImage = input_TensorImage.requires_grad_()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                self.handlers.append(module.register_backward_hook(self.relu_backward_hook))

        output = self.model(self.input_TensorImage)

        if target_label is None:
            target_tensor = torch.zeros_like(output)
            target_tensor[0][torch.argmax(output)] = 1.
        else:
            if isinstance(target_label, int):
                target_tensor = torch.zeros_like(output)
                target_tensor[0][target_label] = 1.
            elif isinstance(target_label, torch.Tensor):
                # if not target_label.dim() == output.dim():
                #     raise NotImplementedError('Dimension of output and target label are different')
                target_tensor = torch.zeros_like(output)
                target_tensor[0][target_label.item()] = 1.

        #  当反向传播作用于一个向量而不是标量时，需要传入一个与其形状相同的权重向量进行加权求和得到一个标量
        #  在可视化任务中，通常目标张量（标签）是最佳选择
        output.backward(target_tensor)

        for handle in self.handlers:
            handle.remove()

        self.guided_gradient = self.input_TensorImage.grad.clone()
        self.input_TensorImage.grad.zero_()
        self.guided_gradient.detach()
        self.guided_gradient = self.normalization(self.guided_gradient)
        return self.guided_gradient
