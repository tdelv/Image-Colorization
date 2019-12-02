import torch
import itertools

def load_data(train_dataset="", batch_size=100, shuffle=True):
	imagenet_data = torchvision.datasets.ImageNet('./hymenoptera_data')
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
	img_batch_lab :: Tensor(batches, 3, height, width) - in LAB color format
	"""

def lab_to_inputs(img_batch_lab):
	"""
	Parameters:
	img_batch_lab :: Tensor(batches, 3, height, width) - in LAB color format

	Returns:
	inputs :: Tensor(batches, 1, height, width) - in L format (just brightness)
	"""

	return img_batch_lab[:, :1, ;, :]

def generate_global_hints(img_batch_lab):
	"""
	Parameters:
	img_batch_lab :: Tensor(batches, 3, height, width) - in LAB color format

	Returns:
	global_hints :: Tensor(batches, 316, 1, 1)
	"""



def generate_local_hints(img_batch_lab):
	"""
	Parameters:
	img_batch_lab :: Tensor(batches, 3, height, width) - in LAB color format

	Returns:
	hints :: Tensor(batches, 2, height, width)
	mask :: Tensor(batches, 1, height, width)
	"""
	# Exponential distribution for how many hints to give with parameter ??? (mentioned in video we watched)
	# Choose that many pixels randomly uniformly (List of pixel, position pairs)

def generate_label(img_batch_lab):
	"""
	Parameters:
	img_batch_lab :: Tensor(batches, 3, height, width) - in LAB color format

	Returns:
	labels :: Tensor(batches, 2, height, width)
	"""

	return img_lab[:, 1;, ;, :]
