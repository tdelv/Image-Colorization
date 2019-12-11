import torch
import skimage
from skimage.transform import resize
import numpy as np
import glob

def load_data(args):
    train_dataset = glob.glob(args.test_dir + "*.JPEG")
    image_collection = skimage.io.ImageCollection(
                        train_dataset, 
                        load_func=image_loader(args), 
                        conserve_memory=not(args.no_conserve_memory))
    train_loader = torch.utils.data.DataLoader(
                        image_collection, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        pin_memory=args.use_gpu, 
                        num_workers=args.num_workers, 
                        drop_last=True)
    num_batches = len(train_dataset) // args.batch_size

    return train_loader, num_batches

def image_loader(args):
    def loader(url):
        img = skimage.io.imread(url) # slow
        img = skimage.util.img_as_float32(img)
        img = resize(img, (args.im_height, args.im_width))

        # handle gray color images
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        
        img_lab = imagenet_to_lab(img, args) # slow
        inputs = generate_input(img_lab, args)
        global_hints = generate_global_hints(img_lab, img, args) # slow
        local_hints, local_mask = generate_local_hints(img_lab, args)
        labels = generate_label(img_lab, args)

        return inputs, global_hints, local_hints, local_mask, labels, url

    return loader

def imagenet_to_lab(img, args):
    """
    Given a batch of images from ImageNet, convert them into a tensor.
    Parameters:
    img :: Tensor(batch_size, height, width, 3) - in RGB color format

    Returns:
    img_batch_lab :: Tensor(batch_size, height, width, 3) - in LAB color format
    """
    
    return torch.from_numpy(skimage.color.rgb2lab(img))


def generate_input(img_lab, args):
    """
    Parameters:
    img_batch_lab :: Tensor(height, width, 3) - in LAB color format

    Returns:
    inputs :: Tensor(height, width, 1) - in L format (just brightness)
    """

    return img_lab[:, :, :1]

# Load color bins
pts_in_hull = np.load('data/pts_in_hull.npy')

def generate_global_hints(img_lab, img_rgb, args):
    """
    Parameters:
    img_batch_lab :: Tensor(height, width, 3) - in LAB color format

    Returns:
    global_hints :: Tensor(1, 1, 316)
    """

    # Generate global hint tensor
    global_hint = torch.zeros(1, 1, 316).float()

    # Color distribution hint
    global_hint[0, 0, 313] = 1.
    
    # Get flattened color array
    ab = img_lab[::4, ::4, 1:].reshape((-1, 2 )).numpy()

    for col in ab:
        # For each color, find distance to each bin center
        dists = pts_in_hull - col
        dists = dists ** 2
        dists = np.sum(dists, axis=1)

        # Find smalleset, and increase bin frequency by 1
        idx = np.argmin(dists)
        global_hint[0, 0, idx] += 16.

    # Saturation hint
    global_hint[0, 0, 314] = 1.
    hsv = skimage.color.rgb2hsv(img_rgb)
    global_hint[0, 0, 315] = float(np.mean(hsv[:, :, 1]))

    return global_hint 

def generate_local_hints(img_lab, args):
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

    for hint in range(10):
        cy, cx = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.tensor([height/2, width/2]), 
            torch.tensor([[(height/4)**2, 0], [0, (width/4)**2]])).sample()
        cy, cx = int(cy), int(cx)
        cy, cx = max(min(cy, args.im_height - 1), 0), max(min(cx, args.im_width - 1), 0)

        size = int(torch.distributions.uniform.Uniform(0, 5).sample())
        lower_y, upper_y = max(0, cy - size), min(cy + size + 1, args.im_height - 1)
        lower_x, upper_x = max(0, cx - size), min(cx + size + 1, args.im_width - 1)

        hints[lower_y:upper_y, lower_x:upper_x, :] = \
            torch.mean(torch.mean(img_lab[lower_y:upper_y, lower_x:upper_x, 1:], 0), 0)
        mask[lower_y:upper_y, lower_x:upper_x, :] = 1.

    return hints, mask

def generate_label(img_lab, args):
    """
    Parameters:
    img_batch_lab :: Tensor(height, width, 3) - in LAB color format

    Returns:
    labels :: Tensor(height, width, 2)
    """

    return img_lab[:, :, 1:]
