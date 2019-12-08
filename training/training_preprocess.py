import torch
import torchvision
import itertools

def load_data(train_dataset="", batch_size=100, shuffle=True):
    imagenet_data = torchvision.datasets.ImageNet('data/ILSVRC2014_train_0000/')
    train_loader = torch.utils.data.DataLoader(
        imagenet_data, batch_size=batch_size, shuffle=shuffle)

    train_loader_lab = map(imagenet_to_lab, img)
    loader1, loader2, loader3, loader4 = itertools.tee(train_loader_lab, 4)

    train_loader_inputs = map(lab_to_bw, loader1)
    train_loader_global_hints = map(generate_global_hints, loader2)
    train_loader_local_hints = map(generate_local_hints, loader3)
    train_loader_labels = map(generate_label, loader4)

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

def imagenet_to_lab(img):
    """
    Given a batch of images from ImageNet, convert them into a tensor.
    Parameters:
    ????????????????

    Returns:
    img_batch_lab :: Tensor(batch_size, 3, height, width) - in LAB color format
    """

def lab_to_inputs(img_batch_lab):
    """
    Parameters:
    img_batch_lab :: Tensor(batch_size, 3, height, width) - in LAB color format

    Returns:
    inputs :: Tensor(batch_size, 1, height, width) - in L format (just brightness)
    """

    return img_batch_lab[:, :1, :, :]

def generate_global_hints(img_batch_lab):
    """
    Parameters:
    img_batch_lab :: Tensor(batch_size, 3, height, width) - in LAB color format

    Returns:
    global_hints :: Tensor(batch_size, 316, 1, 1)
    """

    pass

def generate_local_hints(img_batch_lab):
    """
    Parameters:
    img_batch_lab :: Tensor(batch_size, 3, height, width) - in LAB color format

    Returns:
    hints :: Tensor(batch_size, 2, height, width)
    mask :: Tensor(batch_size, 1, height, width)
    """

    batch_size, _, height, width = img_batch_lab.shape

    hints = torch.zeros(batch_size, 2, height, width)
    mask = torch.zeros(batch_size, 1, height, width)

    for img in range(batch_size):
        reveal_all = int(torch.distributions.bernoulli.Bernoulli(1/100).sample())
        if reveal_all:
            hints[img, :, :, :] = img_batch_lab[img, :, :, :]
            mask[img, :, :, :] = 1.
        else:
            num_hints = int(torch.distributions.geometric.Geometric(1/8).sample())
            for hint in range(num_hints):
                cx, cy = torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.tensor([height/2, width/2]), 
                    torch.tensor([[(height/4)**2, 0], [0, (width/4)**2]])).sample()
                cx, cy = int(cx), int(cy)

                size = int(torch.distributions.uniform.Uniform(0, 5))
                lower_y, upper_y = cy - size, cy + size + 1
                lower_x, upper_x = cx - size, cx + size + 1

                hints[img, :, lower_y:upper_y, lower_x:upper_x] = \
                    torch.mean(img_batch_lab[img, :, lower_y:upper_y, lower_x:upper_x])
                mask[img, :, lower_y:upper_y, lower_x:upper_x] = 1.

    return hints, mask

def generate_label(img_batch_lab):
    """
    Parameters:
    img_batch_lab :: Tensor(batch_size, 3, height, width) - in LAB color format

    Returns:
    labels :: Tensor(batch_size, 2, height, width)
    """

    return img_lab[:, 1:, :, :]
