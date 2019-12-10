from training.training import train, load_model
from training.training_preprocess import image_loader
import warnings
import argparse
import torch
import glob
import skimage.io

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

parser = argparse.ArgumentParser(description='DCGAN')

parser.add_argument('--train-dir', type=str, default='training/data/ILSVRC2014_DET_train/ILSVRC2013_DET_train_extra/',
                    help='Data where training images live')

parser.add_argument('--no-shuffle', action='store_false',
                    help='Shuffle the data when training?')

parser.add_argument('--no-conserve-memory', action='store_false',
                    help='Only load one image a time in memory?')

parser.add_argument('--im-height', type=int, default=64,
                    help='Height of training image rescale.')

parser.add_argument('--im-width', type=int, default=64,
                    help='Width of training image rescale.')

parser.add_argument('--test-dir', type=str, default='data/inputs/',
                    help='Data where testing images live')

parser.add_argument('--out-dir', type=str, default='data/outputs/',
                    help='Data where sampled output images will be written')

parser.add_argument('--mode', type=str, default='train',
                    help='Can be "train" or "test"')

parser.add_argument('--reset-checkpoint', action='store_true',
                    help='Use this flag if you want to resuming training from a previously-saved checkpoint')

parser.add_argument('--batch-size', type=int, default=128,
                    help='Sizes of image batches fed through the network')

parser.add_argument('--num-workers', type=int, default=0,
                    help='Number of threads to use when loading & pre-processing training images')

parser.add_argument('--num-epochs', type=int, default=10,
                    help='Number of passes through the training data to make before stopping')

parser.add_argument('--learn-rate', type=float, default=0.0001,
                    help='Learning rate for Adam optimizer')

parser.add_argument('--beta1', type=float, default=0.9,
                    help='"beta1" parameter for Adam optimizer')

parser.add_argument('--log-every', type=int, default=100,
                    help='Print losses after every [this many] training iterations')

parser.add_argument('--save-every', type=int, default=500,
                    help='Save the state of the network after every [this many] training iterations')

parser.add_argument('--use-gpu', type=bool, default=torch.cuda.is_available(),
                    help='Should we use gpu')

args = parser.parse_args()

if args.mode == 'train':
    train(args)
elif args.mode == 'test':
    model, _, _ = load_model(args=args)
    loader = image_loader(args)
    for file in glob.glob('data/inputs/*.JPEG'):
        inp, gl, loh, lom, lab = loader(file)
        out = model(inp, gl, loh, lom)[0][0]
        out = torch.cat((inp, out.double()), dim=-1).detach().numpy()
        print(out)
        out = skimage.color.lab2rgb(out)
        skimage.io.imsave(file.replace('inputs', 'outputs'), out)
else:
    raise ValueError('--mode should be one of "train" or "test".')

# Check dependencies
# Setup PYTHONFILE
# Check flag for ui
# Setup REPL
# Load model
