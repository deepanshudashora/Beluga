from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer,scheduler,train_losses,train_acc,criterian=nn.CrossEntropyLoss()):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    #loss = F.cross_entropy(pred, target)
    loss = criterian(pred,target)
    train_loss+=loss.item()
    

    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))
  return train_losses,train_acc