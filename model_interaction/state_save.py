
def save_model_training(model, optimizer, epoch):

	torch.save({'epoch': epoch,
	            'model_state_dict': model.state_dict(),
	            'optimizer_state_dict': optimizer.state_dict()}, 
	            "save_states/state-epoch{epoch}.tar".format(epoch=epoch))


def load_model_training(model, optimizer, epoch=None):
	checkpoint = torch.load(PATH)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	model.train()