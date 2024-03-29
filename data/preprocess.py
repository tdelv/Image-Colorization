import torch
from skimage import io

def jpeg_to_array(filepath):
    """
    Parameters:
    img :: file path to jpeg image

    Returns:
    bw_pixels :: MxN np array of pixels of black and white version of image
    color_pixels :: MxNx3 np array of pixels of image
    """
    return io.imread(filepath, as_gray = True), io.imread(filepath)


def bw_image_to_tensor(img, type="RGB"):
    """
    Parameters:
    img :: Image - Not sure what format this is in for Python
    type :: String - The type of image (RGB vs. L, etc.)

    Returns:
    pixels :: Tensor(1, height, width)
    """

    if type == "RGB":
        pass
    elif type == "L":
        pass
    else:
        raise "Invalid image type"



def local_hints_to_tensor(hints=[]):
    """
    Parameters:
    hints :: List of tuples ((x, y), (a, b)) - for hinting position (x, y) is color (a, b)

    Returns:
    hints :: Tensor(2, height, width) - representing the ab hint values;
             0, 0 for non-provided pixels
    mask :: Tensor(1, height, width) - representing which pixels are hinted
    """

    # TODO: get the image shape
    h = -1 # temporary value
    w = -1 # temporary value

    hints_tensor = torch.zeros([2, h, w], dtype=torch.int32)
    mask_tensor = torch.zeros([1, h, w], dtype=torch.int32)

    for hint in hints:
        # extracting x, y, a, b from the current hint
        x, y = hint[0]
        a, b = hint[1]
        # storing the relevant info in the hints tensor
        hints_tensor[0][x][y] = a
        hints_tensor[1][x][y] = b
        # marking this hint in the mask tensor
        mask_tensor[0][x][y] = 1

    return hints_tensor, mask_tensor


def global_hint_to_tensor(img=None, type="RGB", saturation=None):
    """
    Parameters:
    img :: Image - Not sure what format this is in for Python
    type :: String - The type of image (RGB probably?)
    saturation :: Number - The saturation hint

    Returns:
    global_hint :: Tensor(316, 1, 1) - in the following order:
                   - global hint
                   - saturation hint
                   - global hint indicator
                   - saturation hint indicator
    """
    # Turn color image RGB into LAB
    # Turn color image LAB into color bin distribution tensor (313, 1, 1)
    # Turn saturation into tensor (1, 1, 1)
    # Turn this into global hint tensor (313 + 1 + 1 + 1 = 316, 1, 1)
    # The extra two values represent whether each of these values was actually provided, 1 for yes, 0 for no.
    # Just concatenate.

    pass
