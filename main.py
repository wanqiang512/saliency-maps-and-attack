import os
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Normalize
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import argparse
from torchvision.models import (
    resnet18, resnet50, vgg19, inception_v3, alexnet, densenet121, swin_t, swin_b, swin_s,
    ResNet18_Weights, ResNet50_Weights, Inception_V3_Weights, AlexNet_Weights, Swin_B_Weights, Swin_T_Weights,
    Swin_S_Weights, DenseNet121_Weights, VGG19_Weights
)

parser = argparse.ArgumentParser(description="Test.py")
parser.add_argument('--root', type=str, default=r'D:\code\CAM-attack\dataset\ImageNetval_5000\val5000',
                    help='Input images.')
parser.add_argument('--save_adv', type=str, default='adv_img/', help='Output directory with adv images.')
parser.add_argument('--modeltype', type=str, default='resnet18', help='Substitution model.')
opt = parser.parse_args()


def load_model(model_name):
    if model_name == 'resnet18':
        return resnet18(weights=ResNet18_Weights.DEFAULT)
    if model_name == 'resnet50':
        return resnet50(weights=ResNet50_Weights.DEFAULT)
    elif model_name == 'densenet121':
        return densenet121(weights=DenseNet121_Weights.DEFAULT)
    elif model_name == 'vgg19':
        return vgg19(weights=VGG19_Weights.DEFAULT)
    elif model_name == 'Inc-v3':
        return inception_v3(weights=Inception_V3_Weights.DEFAULT)
    elif model_name == 'alexnet':
        return alexnet(weights=AlexNet_Weights.DEFAULT)
    elif model_name == 'swin_t':
        return swin_t(weights=Swin_T_Weights.DEFAULT)
    elif model_name == 'swin_b':
        return swin_b(weights=Swin_B_Weights.DEFAULT)
    elif model_name == 'swin_s':
        return swin_s(weights=Swin_S_Weights.DEFAULT)
    else:
        print('Not supported model')


def mkdir(path):
    """Check if the folder exists, if it does not exist, create it"""
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def save_image(images, filenames, output_dir):
    """save high quality jpeg"""
    mkdir(output_dir)
    for i, filename in enumerate(filenames):
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save(os.path.join(output_dir, filename))


def run_attack(
        method: str,
        use_Inc_model: bool = False,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        save_img: bool = False
):
    if use_Inc_model:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(299),
                transforms.CenterCrop(299),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ]
        )
    print('Loaded source model...')
    model = load_model(opt.modeltype)
    model = torch.nn.Sequential(
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model.eval().to(device)
    )

    model_name = opt.modeltype

    print('Loaded transfer models...')
    all_model_names = ['resnet18', 'resnet50', 'densenet121', 'vgg19', 'Inc-v3', 'alexnet']
    transfer_model_names = [x for x in all_model_names if x != opt.modeltype]
    transfer_models = [load_model(x) for x in transfer_model_names]
    for model_ in transfer_models:
        model_ = torch.nn.Sequential(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            model_.eval().to(device)
        )

    success_rate = dict()  # 攻击成功率
    for name in all_model_names:
        success_rate[name] = 0
    print('Loaded dataset...')
    val_dataset = ImageFolder(root=opt.root, transform=transform)  # root = "mini-imagenet or other"
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=0)
    print('Image Loaded...')
    for batch, (images, labels) in enumerate(tqdm(val_loader)):
        images = images.to(device)
        labels = labels.to(device)

        if method == "test":
            from Attack.IDG import TAIDG
            attack = TAIDG()
            adv = attack(model, images, labels)

        output = model(adv).max(dim=1)[1]
        success_rate[model_name] += (output != labels).sum().item()

        for transfer_model_name, transfer_model in zip(transfer_model_names, transfer_models):
            output = transfer_model(adv).max(dim=1)[1]
            success_rate[transfer_model_name] += (output != labels).sum().item()

        # save adv image ?
        if save_img:
            filename = [str(batch * len(labels) + _) + ".jpg" for _ in range(len(images))]
            save_image(images=adv, filenames=filename, output_dir="./adv_img")

    for model_name_ in success_rate.keys():
        print('Model: %s attack Success Rate:%f' % (model_name_, success_rate[model_name_] / len(val_dataset)))


if __name__ == '__main__':
    run_attack(method="test", use_Inc_model=False, save_img=True)
