import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import cv2
from torchvision import datasets
from torch_lr_finder import LRFinder
from custom_models.gradcam_utils import GradCAM
# Data to plot accuracy and loss graphs
import uuid 
import os
os.makedirs("logs/", exist_ok=True)

import logging
logging.basicConfig(filename='logs/network.log', format='%(asctime)s: %(filename)s: %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}


def unnormalize(img,mean,std):
    img = img.cpu().numpy().astype(dtype=np.float32)
  
    for i in range(img.shape[0]):
        img[i] = (img[i]*std[i])+mean[i]
    
    return np.transpose(img, (1,2,0))


def implement_onecycle_policy(model_class,configuration,device,train_loader):
    learning_rate = configuration.get('learning_rate')
    weight_decay = configuration.get('weight_decay')
    end_lr = configuration.get('end_lr')
    num_iterations = configuration.get('num_iterations')
    step_mode = configuration.get('step_mode')
    
    
    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iterations, step_mode=step_mode)
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state


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


def plot_misclassified(model, test_loader,test_data, device,mean,std,grid_size,no_misclf=20, title='Misclassified'):
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
  
  
  rows, cols = grid_size[0], grid_size[1]
  figure = plt.figure(figsize=(10,14))

  for i in range(1, cols * rows + 1):
    img, label, pred = misclf[i-1]

    figure.add_subplot(rows, cols, i) # adding sub plot
    plt.suptitle(title, fontsize=10)
    plt.title(f"Pred label: {classes[pred]}\n True label: {classes[label]}") # title of plot
    plt.axis("off") # hiding the axis
    img = img.squeeze().numpy()
    img = np.transpose(img, (1, 2, 0))
    name = str(uuid.uuid4())
    cv2.imwrite(f"missclassified/missclassified_image_{label}_{name}.png",img)
    plt.imshow(img, cmap="gray") # showing the plot

  plt.show()
  return misclf

def plot_trueclassified(model, test_loader,test_data, device,mean,std,grid_size,no_clf=20, title='Misclassified'):
  count = 0
  k = 30
  clf = list()
  classes = test_data.classes
  
  while count<=no_clf:
    img, label = test_loader.dataset[k]
    pred = model(img.unsqueeze(0).to(device)) # Prediction
    # pred = model(img.unsqueeze(0).to(device)) # Prediction
    pred = pred.argmax().item()

    k += 1
    if pred==label:
      denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))
      img = denormalize(img)
      clf.append((img, label, pred))
      count += 1
  
  
  rows, cols = grid_size[0], grid_size[1]
  figure = plt.figure(figsize=(10,14))

  for i in range(1, cols * rows + 1):
    img, label, pred = clf[i-1]

    figure.add_subplot(rows, cols, i) # adding sub plot
    plt.suptitle(title, fontsize=10)
    plt.title(f"Pred label: {classes[pred]}\n True label: {classes[label]}") # title of plot
    plt.axis("off") # hiding the axis
    img = img.squeeze().numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img, cmap="gray") # showing the plot

  plt.show()
  return clf



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
  
def generate_gradcam(misclassified_images, model, target_layers,device):
    images=[]
    labels=[]
    for i, (img, pred, correct) in enumerate(misclassified_images):
        images.append(img)
        labels.append(correct)
    
    model.eval()
    
    # map input to device
    images = torch.stack(images).to(device)
    
    # set up grad cam
    gcam = GradCAM(model, target_layers)
    
    # forward pass
    probs, ids = gcam.forward(images)
    
    # outputs agaist which to compute gradients
    ids_ = torch.LongTensor(labels).view(len(images),-1).to(device)
    
    # backward pass
    gcam.backward(ids=ids_)
    layers = []
    for i in range(len(target_layers)):
        target_layer = target_layers[i]
        print("Generating Grad-CAM @{}".format(target_layer))
        # Grad-CAM
        layers.append(gcam.generate(target_layer=target_layer))
        
    # remove hooks when done
    gcam.remove_hook()
    return layers, probs, ids

def plot_gradcam(gcam_layers, target_layers, class_names, image_size,predicted, misclassified_images,mean,std):
    
  images=[]
  labels=[]
  for i, (img, pred, correct) in enumerate(misclassified_images):
    images.append(img)
    labels.append(correct)

  c = len(images)+1
  r = len(target_layers)+2
  fig = plt.figure(figsize=(40,20))
  fig.subplots_adjust(hspace=0.02, wspace=0.02)
  ax = plt.subplot(r, c, 1)
  ax.text(0.3,-0.5, "INPUT", fontsize=14)
  plt.axis('off')
  for i in range(len(target_layers)):
    target_layer = target_layers[i]
    ax = plt.subplot(r, c, c*(i+1)+1)
    ax.text(0.3,-0.5, target_layer, fontsize=14)
    plt.axis('off')

    for j in range(len(images)):
      img = np.uint8(255*unnormalize(images[j].view(image_size),mean,std))
      if i==0:
        ax = plt.subplot(r, c, j+2)
        ax.text(0, 0.4, f"actual: {class_names[labels[j]]} \npredicted: {class_names[predicted[j][0]]}", fontsize=12)
        plt.axis('off')
        plt.subplot(r, c, c+j+2)
        plt.imshow(img)
        plt.axis('off')


      heatmap = 1-gcam_layers[i][j].cpu().numpy()[0] # reverse the color map
      heatmap = np.uint8(255 * heatmap)
      heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
      superimposed_img = cv2.resize(cv2.addWeighted(img, 0.5, heatmap, 0.5, 0), (128,128))
      plt.subplot(r, c, (i+2)*c+j+2)
      plt.imshow(superimposed_img, interpolation='bilinear')

      plt.axis('off')
  plt.show()