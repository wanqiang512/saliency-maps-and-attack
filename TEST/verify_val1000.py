"""Implementation of evaluate attack result."""
import glob
import os
import numpy as np
import torch
from imageio.v2 import imread
from torch import nn
from tqdm import tqdm
from Normalize import Normalize, TfNormalize
import pretrainedmodels
import argparse

opt = argparse.ArgumentParser()
opt.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
opt.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
opt.add_argument("--batch_size", type=int, default=20, help="How many images process at one time.")
opt.add_argument('--adv_dir', type=str, default='./checkpoint/outputs/')
opt.add_argument("--label_file", type=str, default='../dataset/imagenet val_1000/val_rs.csv')
FLAGS = opt.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'inception_v3':
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                    pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'inception_v4':
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                    pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_v2_50':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_v2_101':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    pretrainedmodels.resnet101(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_v2_152':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    pretrainedmodels.resnet152(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'inc_res_v2':
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                    pretrainedmodels.inceptionresnetv2(num_classes=1000,
                                                                       pretrained='imagenet').eval().cuda())
    elif net_name == 'tf2torch_adv_inception_v3':
        from torch_nets import tf_adv_inception_v3
        net = tf_adv_inception_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(), )
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        from torch_nets import tf_ens3_adv_inc_v3
        net = tf_ens3_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(), )
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        from torch_nets import tf_ens4_adv_inc_v3
        net = tf_ens4_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(), )
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        from torch_nets import tf_ens_adv_inc_res_v2
        net = tf_ens_adv_inc_res_v2
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(), )
    else:
        print('Wrong model name!')

    return model


def load_images(input_dir, batch_shape):
    f2l = load_labels(FLAGS.label_file)
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    labels = []
    batch_size = batch_shape[0]
    # for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
    for filepath in glob.glob(os.path.join(input_dir, '*')):
        # with tf.gfile.Open(filepath, 'rb') as f:
        with open(filepath, 'rb') as f:
            image = imread(f, pilmode='RGB').astype(float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        # images[idx, :, :, :] = image * 2.0 - 1.0
        images[idx, :, :, :] = image
        filenames.append(os.path.basename(filepath))
        idx += 1
        labels.append(f2l[os.path.basename(filepath)])
        imgs = torch.from_numpy(images.transpose(0, 3, 1, 2)).to(torch.float32)
        labs = torch.from_numpy(np.array(labels)).to(torch.long)
        if idx == batch_size:
            yield filenames, imgs, labs
            filenames = []
            images = np.zeros(batch_shape)
            labels = []
            idx = 0
    if idx > 0:
        yield filenames, images


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] - 1 for i in range(len(dev))}
    return f2l


def verify(model_name, path):
    img_size = 299
    total_batches = len(glob.glob(os.path.join(FLAGS.adv_dir, '*'))) // FLAGS.batch_size
    model = get_model(model_name, path)
    sum = 0
    for filenames, images, labels in tqdm(
            load_images(FLAGS.adv_dir, [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]),
            desc="evaluate attack ...", total=total_batches
    ):
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            sum += (model(images).argmax(1) != (labels)).detach().sum().cpu()
    print("Attack Success Rate for {0} : {1:.1f}%".format(model_name, sum / 1000. * 100))


def verify_ensmodels(model_name, path):
    img_size = 299
    total_batches = len(glob.glob(os.path.join(FLAGS.adv_dir, '*'))) // FLAGS.batch_size
    model = get_model(model_name, path)
    sum = 0
    for filenames, images, labels in tqdm(
            load_images(FLAGS.adv_dir, [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]),
            desc="evaluate attack ...", total=total_batches
    ):
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            sum += (model(images).argmax(1) != (labels + 1)).detach().sum().cpu()
    print("Attack Success Rate for {0} : {1:.1f}%".format(model_name, sum / 1000. * 100))


def main():
    model_names = ['inception_v3', 'inception_v4', 'resnet_v2_50']
    model_names_ens = ['tf2torch_ens3_adv_inc_v3',
                       'tf2torch_ens4_adv_inc_v3']  # You can download the pretrained ens_models from https://github.com/ylhz/tf_to_pytorch_model
    models_path = './torch_nets_weight/'
    for model_name in model_names:
        verify(model_name, models_path)
        print("===================================================")
    for model_name in model_names_ens:  # When we validate the ens model, we should change gt to gt+1 as the ground truth label.
        verify_ensmodels(model_name, models_path)
        print("===================================================")


if __name__ == '__main__':
    print(FLAGS.adv_dir)
    main()
