import torch
import torchvision
import data.preprocess
import skimage
from skimage.transform import resize
import itertools
from numpy.random import randint
import numpy as np
import time

IM_HEIGHT, IM_WIDTH = 64, 64


def load_data(train_dataset="training/data/*/*.JPEG", batch_size=100, shuffle=True):
    image_collection = skimage.io.ImageCollection(train_dataset, load_func=image_loader, conserve_memory=True)
    train_loader = torch.utils.data.DataLoader(
        image_collection, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)

    return train_loader

    '''
    train_loader_lab = map(imagenet_to_lab, train_loader)
    loader1, loader2, loader3, loader4 = itertools.tee(train_loader_lab, 4)

    train_loader_inputs = map(lab_to_inputs, loader1)
    train_loader_global_hints = map(generate_global_hints, loader2)
    train_loader_local_hints = map(generate_local_hints, loader3)
    train_loader_labels = map(generate_label, loader4)

    return train_loader_inputs, \
           train_loader_global_hints, \
           train_loader_local_hints, \
           train_loader_labels
    '''

def image_loader(url):
    start_time  = time.time()
   
    img = skimage.io.imread(url) # slow
    img = skimage.util.img_as_float32(img)
    img = resize(img, (IM_HEIGHT, IM_WIDTH))

    # handle gray color images
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)
    
    img_lab = imagenet_to_lab(img) # slow
    inputs = generate_input(img_lab)
    global_hints = generate_global_hints(img_lab) # slow
    local_hints, local_mask = generate_local_hints(img_lab)
    labels = generate_label(img_lab)

    return inputs, global_hints, local_hints, local_mask, labels


def imagenet_to_lab(img):
    """
    Given a batch of images from ImageNet, convert them into a tensor.
    Parameters:
    img :: Tensor(batch_size, height, width, 3) - in RGB color format

    Returns:
    img_batch_lab :: Tensor(batch_size, height, width, 3) - in LAB color format
    """
    
    return torch.from_numpy(skimage.color.rgb2lab(img))


def generate_input(img_lab):
    """
    Parameters:
    img_batch_lab :: Tensor(height, width, 3) - in LAB color format

    Returns:
    inputs :: Tensor(height, width, 1) - in L format (just brightness)
    """

    return img_lab[:, :, :1]

# Load color bins
pts_in_hull = np.load('data/pts_in_hull.npy')

def generate_global_hints(img_lab):
    """
    Parameters:
    img_batch_lab :: Tensor(height, width, 3) - in LAB color format

    Returns:
    global_hints :: Tensor(1, 1, 316)
    """
    # Get flattened color array
    ab = img_lab[::4, ::4, 1:]
    ab = torch.reshape(ab, (-1, 2)).numpy()

    # Generate global hint tensor
    bins = torch.zeros(1, 1, 316)
    for col in ab:
        # For each color, find distance to each bin center
        dists = pts_in_hull - col
        dists = dists ** 2
        dists = np.sum(dists, axis=1)

        # Find smalleset, and increase bin frequency by 1
        idx = np.argmin(dists)
        bins[0, 0, idx] += 16

    return bins

def generate_local_hints(img_lab):
    """
    Parameters:
    img_batch_lab :: Tensor(height, width, 3) - in LAB color format

    Returns:
    hints :: Tensor(height, width, 2)
    mask :: Tensor(height, width, 1)
    """

    height, width, _ = img_lab.shape

    hints = torch.zeros(height, width, 2)
    mask = torch.zeros(height, width, 1)

    reveal_all = int(torch.distributions.bernoulli.Bernoulli(1/100).sample())
    if reveal_all:
        hints[:, :, :] = img_lab[:, :, 1:]
        mask[:, :, :] = 1.
    else:
        num_hints = int(torch.distributions.geometric.Geometric(1/8).sample())
        for hint in range(num_hints):
            cy, cx = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.tensor([height/2, width/2]), 
                torch.tensor([[(height/4)**2, 0], [0, (width/4)**2]])).sample()
            cy, cx = int(cy), int(cx)
            cy, cx = max(min(cy, IM_HEIGHT - 1), 0), max(min(cx, IM_WIDTH - 1), 0)

            size = int(torch.distributions.uniform.Uniform(0, 5).sample())
            lower_y, upper_y = max(0, cy - size), min(cy + size + 1, IM_HEIGHT - 1)
            lower_x, upper_x = max(0, cx - size), min(cx + size + 1, IM_WIDTH - 1)

            hints[lower_y:upper_y, lower_x:upper_x, :] = \
                torch.mean(torch.mean(img_lab[lower_y:upper_y, lower_x:upper_x, 1:], 0), 0)
            mask[lower_y:upper_y, lower_x:upper_x, :] = 1.

    return hints, mask

def generate_label(img_lab):
    """
    Parameters:
    img_batch_lab :: Tensor(height, width, 3) - in LAB color format

    Returns:
    labels :: Tensor(height, width, 2)
    """

    return img_lab[:, :, 1:]
