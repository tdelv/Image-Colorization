

def bw_image_to_tensor(img, type="RGB"):
    """
    Parameters:
    img :: Image - Not sure what format this is in for Python
    type :: String - The type of image (RGB vs. L, etc.)

    Returns:
    pixels :: A tensor of shape (1, H, W)
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
    hints :: A tensor of shape (2, H, W) - representing the ab hint values;
             0, 0 for non-provided pixels
    mask :: A tensor of shape (1, H, W) - representing which pixels are hinted
    """

    pass


def global_hint_to_tensor(img=None, type="RGB", saturation=None):
    """
    Parameters:
    img :: Image - Not sure what format this is in for Python
    type :: String - The type of image (RGB probably?)
    saturation :: Number - The saturation hint

    Returns:
    global_hint :: A tensor of shape (316, 1, 1) - in the following order:
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