import glob
import os
from typing import List
import pretrainedmodels
import torch
import numpy as np
from imageio.v2 import imsave, imread
from tqdm import tqdm
import argparse

opt = argparse.ArgumentParser()
opt.add_argument('--attack_name', type=str, default='mifgsm')
opt.add_argument('--input_dir', type=str, default='../dataset/imagenet val_1000/val_rs')
opt.add_argument('--output_dir', type=str, default='./checkpoints/ceshi/')
opt.add_argument("--batch_size", type=int, default=20, help="How many images process at one time.")
opt.add_argument("--label_file", type=str, default='../dataset/imagenet val_1000/val_rs.csv')
opt.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
opt.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
FLAGS = opt.parse_args()


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
        yield filenames, imgs, labs


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    if output_dir is not None:
        check_or_create_dir(output_dir)
    if isinstance(images, torch.Tensor):
        images = np.transpose(images.detach().cpu().numpy(), (0, 2, 3, 1)) * 255
    for i, filename in enumerate(filenames):
        with open(os.path.join(output_dir, filename), 'wb') as f:
            imsave(f, images[i, :, :, :].astype('uint8'), format='png')


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] - 1 for i in range(len(dev))}
    return f2l


if __name__ == '__main__':
    total_batches = len(glob.glob(os.path.join(FLAGS.input_dir, '*'))) // FLAGS.batch_size
    model = pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet').eval().cuda()
    from EM.ceshi import mifgsm

    attack = mifgsm(model)
    for filenames, images, labels in tqdm(
            load_images(FLAGS.input_dir, [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]),
            desc=f"Load images... attack... {FLAGS.attack_name} ...", total=total_batches
    ):
        images = images.cuda()
        labels = labels.cuda()
        adv = attack(images, labels)
        # save_images(adv_images, filenames, FLAGS.output_dir
        save_images(adv, filenames, FLAGS.output_dir)
