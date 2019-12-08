import training.training_preprocess as tp
import skimage
from skimage.transform import resize
import torch
import matplotlib.pyplot as plt

data = tp.load_data()

for inp, g_hint, l_hint, label in zip(*data):
    '''
    print(inp.shape)
    print(g_hint.shape)
    print((l_hint[0].shape, l_hint[1].shape))
    print(label.shape)
    '''

    for l, ab, lab in zip(inp, l_hint[0], g_hint):
        fig, axs = plt.subplots(1, 2)
        img = torch.cat((l.double(), ab.double()), dim=-1)
        img = skimage.color.lab2rgb(img)
        img = resize(img, (512, 512)) 
        #skimage.io.imsave('~/Desktop/img.JPEG', img)
        axs[0].imshow(img)

        img = lab
        img = skimage.color.lab2rgb(img)
        img = resize(img, (512, 512)) 
        axs[1].imshow(img)
        plt.show()

# Check dependencies
# Setup PYTHONFILE
# Check flag for ui
# Setup REPL
# Load model
