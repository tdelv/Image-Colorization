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

    input((inp.shape, g_hint.shape, l_hint[0].shape, l_hint[1].shape, label.shape))
    input(torch.sum(g_hint[0]))

# Check dependencies
# Setup PYTHONFILE
# Check flag for ui
# Setup REPL
# Load model
