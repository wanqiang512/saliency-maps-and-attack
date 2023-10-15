"""
@author: Britney(wanqiang512)
@software: PyCharm
@file: core.py
@time: 2023/10/15 23:23
"""
import numpy as np
import torch
import torch.nn.functional as F
from statistics import mode, mean

from torch import nn


class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class CAM(object):
    """ Class Activation Mapping """

    def __init__(self, model, target_layer):
        """
        Args:
            model: a base model to get saliency_maps which have global pooling and fully connected layer.
            target_layer: conv_layer before Global Average Pooling
        """

        self.model = model
        self.target_layer = target_layer

        # save values of activations and gradients in target_layer
        self.values = SaveValues(self.target_layer)

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # object classification
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # cam can be calculated from the weights of linear layer and activations
        weight_fc = list(
            self.model._modules.get('fc').parameters())[0].to('cpu').data

        cam = self.getCAM(self.values, weight_fc, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getCAM(self, values, weight_fc, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        weight_fc: the weight of fully connected layer.  shape => (num_classes, C)
        idx: predicted class id
        cam: class activation map.  shape => (1, num_classes, H, W)
        '''

        cam = F.conv2d(values.activations, weight=weight_fc[:, :, None, None])
        _, _, h, w = cam.shape

        # class activation mapping only for the predicted class
        # cam is normalized with min-max.
        cam = cam[:, idx, :, :]
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.view(1, 1, h, w)

        return cam.data


class GradCAM(CAM):
    """ Grad saliency_maps """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)

        """
        Args:
            model: a base model to get saliency_maps, which need not have global pooling and fully connected layer.
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: the index of the target class
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # anomaly detection
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getGradCAM(self.values, score, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getGradCAM(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()

        score[0, idx].backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape
        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data


class GradCAMpp(CAM):
    """ Grad saliency_maps plus plus """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: the index of the target class
        Return:
            heatmap: class activation mappings of predicted classes
        """

        # object classification
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getGradCAMpp(self.values, score, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getGradCAMpp(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax. shape => (1, n_classes)
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()

        score[0, idx].backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape

        # calculate alpha
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2)
        ag = activations * gradients.pow(3)
        denominator += ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
        denominator = torch.where(
            denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)

        relu_grad = F.relu(score[0, idx].exp() * gradients)
        weights = (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (weights * activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data


class SmoothGradCAMpp(CAM):
    """ Smooth Grad saliency_maps plus plus """

    def __init__(self, model, target_layer, n_samples=25, stdev_spread=0.15):
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
            n_sample: the number of samples
            stdev_spread: standard deviationß
        """

        self.n_samples = n_samples
        self.stdev_spread = stdev_spread

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: the index of the target class
        Return:
            heatmap: class activation mappings of predicted classes
        """

        stdev = self.stdev_spread / (x.max() - x.min())
        std_tensor = torch.ones_like(x) * stdev

        indices = []
        probs = []

        for i in range(self.n_samples):
            self.model.zero_grad()

            x_with_noise = torch.normal(mean=x, std=std_tensor)
            x_with_noise.requires_grad_()

            score = self.model(x_with_noise)

            prob = F.softmax(score, dim=1)

            if idx is None:
                prob, idx = torch.max(prob, dim=1)
                idx = idx.item()
                probs.append(prob.item())

            indices.append(idx)

            score[0, idx].backward(retain_graph=True)

            activations = self.values.activations
            gradients = self.values.gradients
            n, c, _, _ = gradients.shape

            # calculate alpha
            numerator = gradients.pow(2)
            denominator = 2 * gradients.pow(2)
            ag = activations * gradients.pow(3)
            denominator += \
                ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
            denominator = torch.where(
                denominator != 0.0, denominator, torch.ones_like(denominator))
            alpha = numerator / (denominator + 1e-7)

            relu_grad = F.relu(score[0, idx].exp() * gradients)
            weights = \
                (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)

            # shape => (1, 1, H', W')
            cam = (weights * activations).sum(1, keepdim=True)
            cam = F.relu(cam)
            cam -= torch.min(cam)
            cam /= torch.max(cam)

            if i == 0:
                total_cams = cam.clone()
            else:
                total_cams += cam

        total_cams /= self.n_samples
        idx = mode(indices)
        prob = mean(probs)

        print("predicted class ids {}\t probability {}".format(idx, prob))

        return total_cams.data

    def __call__(self, x):
        return self.forward(x)


class ScoreCAM(CAM):
    """ Score saliency_maps """

    def __init__(self, model, target_layer, n_batch=32):
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
        """
        self.n_batch = n_batch

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: the index of the target class
        Return:
            heatmap: class activation mappings of predicted classes
        """

        with torch.no_grad():
            _, _, H, W = x.shape
            device = x.device

            self.model.zero_grad()
            score = self.model(x)
            prob = F.softmax(score, dim=1)

            if idx is None:
                p, idx = torch.max(prob, dim=1)
                idx = idx.item()
                # print("predicted class ids {}\t probability {}".format(idx, p))

            # # calculate the derivate of probabilities, not that of scores
            # prob[0, idx].backward(retain_graph=True)

            self.activations = self.values.activations.to('cpu').clone()
            # put activation maps through relu activation
            # because the values are not normalized with eq.(1) without relu.
            self.activations = F.relu(self.activations)
            self.activations = F.interpolate(
                self.activations, (H, W), mode='bilinear')
            _, C, _, _ = self.activations.shape

            # normalization
            act_min, _ = self.activations.view(1, C, -1).min(dim=2)
            act_min = act_min.view(1, C, 1, 1)
            act_max, _ = self.activations.view(1, C, -1).max(dim=2)
            act_max = act_max.view(1, C, 1, 1)
            denominator = torch.where(
                (act_max - act_min) != 0., act_max - act_min, torch.tensor(1.)
            )

            self.activations = self.activations / denominator

            # generate masked images and calculate class probabilities
            probs = []
            for i in range(0, C, self.n_batch):
                mask = self.activations[:, i:i + self.n_batch].transpose(0, 1)
                mask = mask.to(device)
                masked_x = x * mask
                score = self.model(masked_x)
                probs.append(F.softmax(score, dim=1)[:, idx].to('cpu').data)

            probs = torch.stack(probs)
            weights = probs.view(1, C, 1, 1)

            # shape = > (1, 1, H, W)
            cam = (weights * self.activations).sum(1, keepdim=True)
            cam = F.relu(cam)
            cam -= torch.min(cam)
            cam /= torch.max(cam)

        return cam.data

    def __call__(self, x):
        return self.forward(x)


class LayerCAM(CAM):
    def __init__(self, model, target_Layer):
        super().__init__(model, target_Layer)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x, idx=None, ):
        """
        Args:
                x: input image. shape =>(1, 3, H, W)
                idx: the index of the target class
        Return:
                heatmap: class activation mappings of predicted classes
        """
        b, c, h, w = x.size()
        logits = self.model(x)

        if idx is None:
            prob, idx = torch.max(logits, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        one_hot = F.one_hot(idx, len(logits[0]))
        logits = F.softmax(logits, dim=1)
        self.model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=False)
        activation = self.values.activations.clone().detach()
        gradients = self.values.gradients.clone().detach()
        with torch.no_grad():
            activation_maps = activation * F.relu(gradients)
            cam = torch.sum(activation_maps, dim=1, keepdim=True)
            cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
            cam = F.relu(cam)
            cam = cam - cam.min() / (cam.max() - cam.min())
        return cam.data

    def __call__(self, x):
        return self.forward(x)


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


class RISE(nn.Module):
    def __init__(
            self,
            model,
            n_masks=10000,
            p1=0.1,
            input_size=(224, 224),
            initial_mask_size=(7, 7),
            n_batch=128,
            mask_path=None
    ):
        super().__init__()
        self.model = model
        self.n_masks = n_masks
        self.p1 = p1
        self.input_size = input_size
        self.initial_mask_size = initial_mask_size
        self.n_batch = n_batch

        if mask_path is not None:
            self.masks = self.load_masks(mask_path)
        else:
            self.masks = self.generate_masks()

    def generate_masks(self):
        # cell size in the upsampled mask
        Ch = np.ceil(self.input_size[0] / self.initial_mask_size[0])
        Cw = np.ceil(self.input_size[1] / self.initial_mask_size[1])

        resize_h = int((self.initial_mask_size[0] + 1) * Ch)
        resize_w = int((self.initial_mask_size[1] + 1) * Cw)

        masks = []

        for _ in range(self.n_masks):
            # generate binary mask
            binary_mask = torch.randn(
                1, 1, self.initial_mask_size[0], self.initial_mask_size[1])
            binary_mask = (binary_mask < self.p1).float()

            # upsampling mask
            mask = F.interpolate(
                binary_mask, (resize_h, resize_w), mode='bilinear', align_corners=False)

            # random cropping
            i = np.random.randint(0, Ch)
            j = np.random.randint(0, Cw)
            mask = mask[:, :, i:i + self.input_size[0], j:j + self.input_size[1]]

            masks.append(mask)

        masks = torch.cat(masks, dim=0)  # (N_masks, 1, H, W)

        return masks

    def load_masks(self, filepath):
        masks = torch.load(filepath)
        return masks

    def save_masks(self, filepath):
        torch.save(self.masks, filepath)

    def forward(self, x):
        # x: input image. (1, 3, H, W)
        device = x.device

        # keep probabilities of each class
        probs = []
        # shape (n_masks, 3, H, W)
        masked_x = torch.mul(self.masks, x.to('cpu').data)

        for i in range(0, self.n_masks, self.n_batch):
            input = masked_x[i:min(i + self.n_batch, self.n_masks)].to(device)
            out = self.model(input)
            probs.append(torch.softmax(out, dim=1).to('cpu').data)

        probs = torch.cat(probs)  # shape => (n_masks, n_classes)
        n_classes = probs.shape[1]

        # caluculate saliency map using probability scores as weights
        saliency = torch.matmul(
            probs.data.transpose(0, 1),
            self.masks.view(self.n_masks, -1)
        )
        saliency = saliency.view(
            (n_classes, self.input_size[0], self.input_size[1]))
        saliency = saliency / (self.n_masks * self.p1)

        # normalize
        m, _ = torch.min(saliency.view(n_classes, -1), dim=1)
        saliency -= m.view(n_classes, 1, 1)
        M, _ = torch.max(saliency.view(n_classes, -1), dim=1)
        saliency /= M.view(n_classes, 1, 1)
        return saliency.data


class EigenCAM(CAM):
    """EigenCAM  """

    def __init__(self, model, target_layer):
        """
            Args:
                    model: a base model
                    target_layer: conv_layer you want to visualize
        """
        super().__init__(model, target_layer)

    def _check(self, feature):
        if feature.ndim != 4 or feature.shape[2] * feature.shape[3] == 1:
            raise ValueError(f'Got invalid shape of feature map: {feature.shape}, '
                             'please specify another layer to plot heatmap.')

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: the index of the target class
        Return:
            heatmap: class activation mappings of predicted classes
        """
        with torch.no_grad():
            score = self.model(x)
            if idx is None:
                prob, idx = torch.max(score, dim=1)
                idx = idx.item()
                prob = prob.item()
                print("predicted class ids {}\t probability {}".format(idx, prob))
            feature = self.values.activations.clone().detach()
            self._check(feature)
            _, _, vT = torch.linalg.svd(feature)
            v1 = vT[:, :, 0, :][..., None, :]
            cam = feature @ v1.repeat(1, 1, v1.shape[3], 1)
            cam = cam.sum(dim=1)
            cam = cam - cam.min() / (cam.max() - cam.min)
        return cam.data

    def __call__(self, x):
        return self.forward(x)
