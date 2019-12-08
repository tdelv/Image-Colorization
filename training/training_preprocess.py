import torch
import torchvision
import data.preprocess
import skimage
from skimage.transform import resize
import itertools
from numpy.random import randint

IM_HEIGHT, IM_WIDTH = 64, 64


def load_data(train_dataset="training/data/*/*.JPEG", batch_size=100, shuffle=True):
    image_collection = skimage.io.ImageCollection(train_dataset, load_func=image_loader)
    train_loader = torch.utils.data.DataLoader(
        image_collection, batch_size=batch_size, shuffle=shuffle)

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
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    '''

def image_loader(url):
    img = skimage.io.imread(url)
    img = skimage.util.img_as_float32(img)

    '''
    to_pad_height, to_pad_width = max(0, IM_HEIGHT - img.shape[0]), max(0, IM_WIDTH - img.shape[1])
    pad_bound_y, pad_bound_x = to_pad_height // 2, to_pad_width // 2
    pad_bounds_y = (pad_bound_y, to_pad_height - pad_bound_y)
    pad_bounds_x = (pad_bound_x, to_pad_width - pad_bound_x)

    img = skimage.util.pad(img, (pad_bounds_y, pad_bounds_x, (0, 0)))

    to_crop_height, to_crop_width = img.shape[0] - IM_HEIGHT, img.shape[1] - IM_WIDTH
    crop_bound_y, crop_bound_x = int(randint(to_crop_height + 1)), int(randint(to_crop_width + 1))
    crop_bounds_y = (crop_bound_y, to_crop_height - crop_bound_y)
    crop_bounds_x = (crop_bound_x, to_crop_width - crop_bound_x)

    img = skimage.util.crop(img, (crop_bounds_y, crop_bounds_x, (0, 0)))
    '''

    img = resize(img, (IM_HEIGHT, IM_WIDTH))


    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    return img


def imagenet_to_lab(img):
    """
    Given a batch of images from ImageNet, convert them into a tensor.
    Parameters:
    img :: Tensor(batch_size, height, width, 3) - in RGB color format

    Returns:
    img_batch_lab :: Tensor(batch_size, height, width, 3) - in LAB color format
    """
    return torch.from_numpy(skimage.color.rgb2lab(img))


def lab_to_inputs(img_batch_lab):
    """
    Parameters:
    img_batch_lab :: Tensor(batch_size, height, width, 3) - in LAB color format

    Returns:
    inputs :: Tensor(batch_size, height, width, 1) - in L format (just brightness)
    """

    return img_batch_lab[:, :, :, :1]

def generate_global_hints(img_batch_lab):
    """
    Parameters:
    img_batch_lab :: Tensor(batch_size, height, width, 3) - in LAB color format

    Returns:
    global_hints :: Tensor(batch_size, 1, 1, 316)
    """

    return img_batch_lab

def generate_local_hints(img_batch_lab):
    """
    Parameters:
    img_batch_lab :: Tensor(batch_size, height, width, 3) - in LAB color format

    Returns:
    hints :: Tensor(batch_size, height, width, 2)
    mask :: Tensor(batch_size, height, width, 1)
    """

    batch_size, height, width, _ = img_batch_lab.shape

    hints = torch.zeros(batch_size, height, width, 2)
    mask = torch.zeros(batch_size, height, width, 1)

    for img in range(batch_size):
        reveal_all = int(torch.distributions.bernoulli.Bernoulli(1/100).sample())
        if reveal_all:
            hints[img, :, :, :] = img_batch_lab[img, :, :, 1:]
            mask[img, :, :, :] = 1.
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

                hints[img, lower_y:upper_y, lower_x:upper_x, :] = \
                    torch.mean(torch.mean(img_batch_lab[img, lower_y:upper_y, lower_x:upper_x, 1:], 0), 0)
                mask[img, lower_y:upper_y, lower_x:upper_x, :] = 1.

    return hints, mask

def generate_label(img_batch_lab):
    """
    Parameters:
    img_batch_lab :: Tensor(batch_size, height, width, 3) - in LAB color format

    Returns:
    labels :: Tensor(batch_size, height, width, 2)
    """

    return img_batch_lab[:, :, :, 1:]
