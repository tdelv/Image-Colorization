import torch
import itertools

def load_data(train_dataset="", batch_size=100, shuffle=True):
	imagenet_data = torchvision.datasets.ImageNet('./hymenoptera_data')
	train_loader = torch.utils.data.DataLoader(
        imagenet_data, batch_size=batch_size, shuffle=shuffle)

	train_loader_lab = map(imagenet_to_lab, img)
	loader1, loader2, loader3, loader4 = itertools.tee(train_loader_lab, 4)

	train_loader_inputs = map(lab_to_bw, loader1)
	train_loader_local_hints = map(generate_local_hints, loader2)
	train_loader_global_hints = map(generate_global_hints, loader3)
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
	Parameters:

	Returns:
	img_lab :: Image - in LAB color format
	"""


# Create black and white input image
def lab_to_bw(img_lab):
	"""
	Parameters:
	img_lab :: Image - in LAB format from ImageNet

	Returns:
	img_bw :: Image - in L format (just brightness)
	"""


#Turn color image LAB into random local hints AB
def generate_local_hints(img_lab):
	"""
	Parameters:
	img_lab :: Image - in LAB format from ImageNet

	Returns:
	hints :: List of tuples ((x, y), (a, b)) - for hinting position (x, y) is color (a, b)
	"""
	# Exponential distribution for how many hints to give with parameter ??? (mentioned in video we watched)
	# Choose that many pixels randomly uniformly (List of pixel, position pairs)


#Turn color image LAB into random global hints
def generate_global_hints(img_lab):
	"""
	Parameters:
	img_lab :: Image - in LAB format from ImageNet

	Returns:
	global_hint :: Image or None - The image itself or None, randomly
	saturation_hint :: Number or None - The saturation of the image itself or None, randomly
	"""

#Turn color image LAB into label AB (the expected answer)
def generate_label(img_lab):
	"""
	Parameters:
	img_lab :: Image - in LAB format from ImageNet

	Returns:
	label :: List of List of pairs - (a, b)?
	"""
