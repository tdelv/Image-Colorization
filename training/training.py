import os
import re
import torch
from model.model import ColorizationModel
import training.training_preprocess as data
from tqdm import tqdm

def train(epochs):
    '''
    Parameters:
    epochs :: Number - The number of epochs to train.

    Returns:
    model :: ColorizationModel
    '''

    learning_rate = 1e-3

    model = ColorizationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        model = model.cuda()
        print("GPU enabled.")
    else:
        print("GPU not enabled.")

    start_epoch = load_model(model, optimizer)
    end_epoch = start_epoch + epochs
    
    print("Loaded epoch", start_epoch)

    for epoch in range(start_epoch, end_epoch):
        d = data.load_data()
        train_epoch(model, optimizer, d) 
        save_model(model, optimizer, start_epoch + epoch + 1)

    # save_model(model, optimizer, end_epoch)

    return model

def save_model(model, optimizer, epoch):
    '''
    Parameters:
    model :: ColorizationModel - The model to save.
    optimizer :: torch.optim.Adam - The optimizer to save.
    epoch :: Number - The current epoch number.
    '''
    
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, 
                "training/save_states/state-epoch-{epoch}.tar".format(epoch=epoch))


def load_model(model, optimizer, epoch=None):
    '''
    Parameters:
    model :: ColorizationModel - The model to load.
    optimizer :: torch.optim.Adam - The optimizer to load.
    epoch (Optional) :: Number - The epoch number to load. If None, will load highest epoch.

    Returns:
    epoch :: Number - Which epoch was loaded.
    '''

    if epoch == None:
        files = os.listdir("training/save_states")
        matches = map(lambda file: re.search("state-epoch-(.*).tar", file), files)
        well_formed = filter(lambda file: file != None, matches)
        epochs = list(map(lambda s: int(s.group(1)), well_formed))
        if len(epochs) == 0:
            return 0
        epoch = max(epochs)

    checkpoint = torch.load("training/save_states/state-epoch-{epoch}.tar".format(epoch=epoch))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return epoch

def train_epoch(model, optimizer, data): 
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

    print("Begin training epoch")
    prog_bar = tqdm(total=100, desc='Batch', position=0)
    loss_bar = tqdm(total=0, position=1, bar_format='{desc}')
    avg_loss_bar = tqdm(total=0, position=2, bar_format='{desc}')

    model.train()

    total_loss = torch.tensor([0])
    for batch_num, batch in enumerate(data, start=1):
        
        batch = (d.float() for  d in batch)
        if torch.cuda.is_available():
            batch = (d.cuda() for d in batch)

        input_batch, global_hint_batch, local_hint_batch, local_mask_batch, label_batch = batch

        optimizer.zero_grad()

        outputs_batch = model(input_batch, global_hint_batch, local_hint_batch, local_mask_batch)
        loss_batch = loss(outputs_batch[0], label_batch)

        loss_batch.backward()
        optimizer.step()

        loss_val = loss_batch
        total_loss += loss_val
        prog_bar.update(1)
        loss_bar.set_description_str(f'Loss: {loss_val}')
        avg_loss_bar.set_description_str(f'Avg Loss: {total_loss / batch_num}')
    
        if batch_num % 20 == 0:
            print()
            print()
            print()
        
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
    total_loss = torch.sum(img_loss)

    return total_loss
