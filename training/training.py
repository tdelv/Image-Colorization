import os
import re
import torch
from model.model import ColorizationModel
import data.training_preprocess as data

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

    start_epoch = load_model(model, optimizer)
    end_epoch = start_epoch + epochs

    for epoch in range(start_epoch, end_epoch):
        inputs, global_hints, local_hints, labels = data.load_data()
        train_epoch(model, optimizer, inputs, global_hints, local_hints, labels)

    save_model(model, optimizer, end_epoch)

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
                "save_states/state-epoch-{epoch}.tar".format(epoch=epoch))


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
        files = os.listdir("./save_states")
        matches = map(lambda file: re.search("state-epoch-(.*).tar", file), files)
        well_formed = filter(lambda s: file != None, matches)
        epochs = list(map(lambda s: s.group(1), well_formed))
        if len(epochs) == 0:
            return 0
        epoch = max(epochs)

    checkpoint = torch.load("save_states/state-epoch{epoch}.tar".format(epoch=epoch))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return epoch

def train_epoch(model, optimizer, inputs, global_hints, local_hints, labels):
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
    
    model.train()

    for input_batch, local_hint_batch, global_hint_batch, label_batch \
    in zip(inputs, local_hints, global_hints, labels):
        optimizer.zero_grad()

        outputs_batch = model(input_batch, global_hint_batch, local_hint_batch)
        loss_batch = loss(outputs_batch, labels_batch)

        loss_batch.backward()
        optimizer.step()



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

    pixel_loss = (1/2 * torch.pow(diff, 2) * (torch.abs(diff) < delta)) + \
                 (delta * (torch.abs(diff) - 1/2 * delta) * (torch.abs(diff) >= delta))

    img_loss = torch.sum(pixel_loss, (2, 3, 4))
    total_loss = torch.sum(img_loss)

    return total_loss