

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
