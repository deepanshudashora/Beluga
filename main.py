import os
os.makedirs("logs/", exist_ok=True)

import logging
logging.basicConfig(filename='logs/network.log', format='%(asctime)s: %(filename)s: %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

from custom_models.train import train
from custom_models.test import test
import torch.optim as optim
import torch 


test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def fit_model(model,training_parameters,train_loader,test_loader,device):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    optimizer = optim.Adam(model.parameters(), 
                          lr=training_parameters["learning_rate"], 
                          weight_decay=training_parameters["weight_decay"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=training_parameters["max_lr"],
        steps_per_epoch=len(train_loader),
        epochs=training_parameters["num_epochs"],
        pct_start=training_parameters["max_at"],
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear',verbose=False)
    for epoch in range(1, training_parameters["num_epochs"]+1):
        print(f'Epoch {epoch}')
        train_losses,train_acc = train(model, device, train_loader, optimizer,scheduler,train_losses,train_acc)
        test_losses,test_acc = test(model, device, test_loader,test_losses,test_acc)
        # scheduler.step()
        
    logging.info('Training Losses : %s', train_losses)
    logging.info('Training Acccuracy : %s', train_acc)
    logging.info('Test Losses : %s', test_losses)
    logging.info('Test Accuracy : %s', test_acc)
        
    return train_losses, test_losses, train_acc, test_acc

def get_device():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    logger.info("device: %s" % device)
    return device

