
def train(from_epoch, for_epochs):
    pass
    

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