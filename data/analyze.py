import os
import re
import torch
from data.analysis_preprocess import load_data
from tqdm import tqdm
import csv
import skimage.io
import skimage
from model.model import ColorizationModel

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def analyze(args):
    '''
    Parameters:
    epochs :: Number - The number of epochs to train.

    Returns:
    model :: ColorizationModel
    '''

    model_generator = generate_models(args)

    for model, epoch in model_generator:
        analyze_model(model, epoch, args)

    return model

def generate_models(args):
    files = os.listdir("training/save_states")
    matches = map(lambda file: re.search("^state-epoch-(.*).tar", file), files)
    well_formed = filter(lambda file: file != None, matches)
    epochs = list(filter(lambda a: a >= args.num_epochs, sorted(map(lambda s: int(s.group(1)), well_formed))))

    model = ColorizationModel()

    def load_model(epoch, model):
        if args.use_gpu:
            checkpoint = torch.load("training/save_states/state-epoch-{epoch}.tar".format(epoch=epoch), map_location=torch.device('cuda'))
        else:
            checkpoint = torch.load("training/save_states/state-epoch-{epoch}.tar".format(epoch=epoch), map_location=torch.device('cpu'))
        
        model.load_state_dict(checkpoint['model_state_dict']) 
       
        if args.use_gpu:
            print('GPU enabled.')
            model = model.cuda()
        else:
            print('GPU not enabled.')

        print('Model loaded from training/save_states/state-epoch-{epoch}.tar'.format(epoch=epoch))

        return model, epoch

    return map(lambda e: load_model(e, model), epochs)


def analyze_model(model, epoch, args): 
    '''
    Trains model for one epoch.

    Arguments:
    model :: ColorizationModel
    optimizer :: torch.optim.Adam
    inputs :: Iterator<Tensor(batch,height,width,1)> - Black and white image input
    local_hints :: Iterator<Tensor(batch, height, width, 3)> - Local hints
    global_hints :: Iterator<Tensor(batch, 1, 1, 316)> - Global hints
    labels :: Iterator<Tensor(batch, height, width, 2)> - Correct AB predictions
    '''

    print(f"Begin analyzing epoch {epoch}.")

    model.eval()

    data, num_batches = load_data(args)
    prog_bar = tqdm(total=num_batches, desc='Batch', position=0)

    total_loss = 0
    for batch_num, batch in enumerate(data, start=1):
        
        input_batch, global_hint_batch, local_hint_batch, local_mask_batch, label_batch, url_batch = batch
        
        label_batch = label_batch.float()

        if args.use_gpu:
            input_batch = input_batch.cuda()
            global_hint_batch = global_hint_batch.cuda()
            local_hint_batch = local_hint_batch.cuda()
            local_mask_batch = local_mask_batch.cuda()
            label_batch = label_batch.cuda()
        
        outputs_batch = model(input_batch, global_hint_batch, local_hint_batch, local_mask_batch)[0]
        
        loss_batch = loss(outputs_batch, label_batch)

        loss_val = float(loss_batch)
        total_loss += loss_val

        prog_bar.update(1)

        if epoch % 10 == 0:
            if not os.path.exists(f'data/outputs/{epoch}'):
                os.mkdir(f'data/outputs/{epoch}')
            for inp, out, url in zip(input_batch, outputs_batch, url_batch):
                out = torch.cat((inp, out.double()), dim=-1).detach().cpu().numpy()
                out = skimage.color.lab2rgb(out)
                skimage.io.imsave(url.replace('inputs', f'outputs/{epoch}'), skimage.img_as_ubyte(out))

    print(f'Epoch {epoch} complete')

    avg_loss = total_loss / num_batches


    with open('data/epoch_loss_data.csv', mode='a') as loss_file:
        loss_writer = csv.writer(loss_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        loss_writer.writerow([epoch, avg_loss])
        
def loss(outputs, labels):
    '''
    Arguments:
    outputs :: Tensor(batch, height, width, 1)
    labels :: Tensor(batch, height, width, 1)

    Returns:
    total_loss :: Tensor()
    '''
    delta = 1

    diff = labels - outputs

    pixel_loss = (1/2 * torch.pow(diff, 2) * (torch.abs(diff) < delta).float()) + \
                 (delta * (torch.abs(diff) - 1/2 * delta) * (torch.abs(diff) >= delta).float())

    img_loss = torch.sum(pixel_loss, (1, 2, 3))
    total_loss = torch.mean(img_loss)

    return total_loss
