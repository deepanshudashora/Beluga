import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from custom_models.train import train
from custom_models.test import test
from torchvision import datasets
# Data to plot accuracy and loss graphs

import os
os.makedirs("logs/", exist_ok=True)

import logging
logging.basicConfig(filename='logs/network.log', format='%(asctime)s: %(filename)s: %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def get_device():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    logger.info("device: %s" % device)
    return device

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

def plot_accuracy_report(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def show_random_results(test_loader,grid_size,model,device):
  cols, rows = grid_size[0],grid_size[1]
  figure = plt.figure(figsize=(20, 20))
  for i in range(1, cols * rows + 1):
      k = np.random.randint(0, len(test_loader.dataset)) # random points from test dataset
    
      img, label = test_loader.dataset[k] # separate the image and label
      img = img.unsqueeze(0) # adding one dimention
      pred=  model(img.to(device)) # Prediction 

      figure.add_subplot(rows, cols, i) # adding sub plot
      plt.title(f"Predcited label {pred.argmax().item()}\n True Label: {label}") # title of plot
      plt.axis("off") # hiding the axis
      plt.imshow(img.squeeze()) # showing the plot

  plt.show()


def plot_misclassified(model, test_loader,test_data, device,mean,std,no_misclf=20, title='Misclassified'):
  count = 0
  k = 30
  misclf = list()
  classes = test_data.classes
  
  while count<=no_misclf:
    img, label = test_loader.dataset[k]
    pred = model(img.unsqueeze(0).to(device)) # Prediction
    # pred = model(img.unsqueeze(0).to(device)) # Prediction
    pred = pred.argmax().item()

    k += 1
    if pred!=label:
      denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))
      img = denormalize(img)
      misclf.append((img, label, pred))
      count += 1
  
  rows, cols = int(no_misclf/4),4
  figure = plt.figure(figsize=(10,14))

  for i in range(1, cols * rows + 1):
    img, label, pred = misclf[i-1]

    figure.add_subplot(rows, cols, i) # adding sub plot
    plt.suptitle(title, fontsize=10)
    plt.title(f"Pred label: {classes[pred]}\n True label: {classes[label]}") # title of plot
    plt.axis("off") # hiding the axis
    img = img.squeeze().numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img, cmap="gray") # showing the plot

  plt.show()


# For calculating accuracy per class
def calculate_accuracy_per_class(model,device,test_loader,test_data):  
  model = model.to(device)
  model.eval()
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
  with torch.no_grad():
      for data in test_loader:
          images, labels = data
          images, labels = images.to(device), labels.to(device)
          outputs = model(images.to(device))
          _, predicted = torch.max(outputs, 1)
          c = (predicted == labels).squeeze()
          for i in range(10):
              label = labels[i]
              class_correct[label] += c[i].item()
              class_total[label] += 1
  final_output = {}
  classes = test_data.classes
  for i in range(len(classes)):
      print()
      print('Accuracy of %5s : %2d %%' % (
          classes[i], 100 * class_correct[i] / class_total[i]))
      final_output[classes[i]] = 100 * class_correct[i] / class_total[i]
  print(final_output)
  original_class = list(final_output.keys())
  class_accuracy = list(final_output.values())
  plt.figure(figsize=(8, 6))
  plt.bar(original_class, class_accuracy)
  plt.xlabel('classes')
  plt.ylabel('accuracy')
  plt.show()