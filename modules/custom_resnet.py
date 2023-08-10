from torchmetrics import Accuracy
import torch 
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F

def apply_normalization(chennels):
  return nn.BatchNorm2d(chennels)


class CustomResnetModule(LightningModule):
    def __init__(self,max_lr,learning_rate,weight_decay,steps_per_epoch,pct_start,epochs):
        super().__init__()
        # Input Block
        drop = 0.0
        self.max_lr = max_lr
        self.steps_per_epoch=steps_per_epoch
        self.epochs=epochs
        self.pct_start=pct_start
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.preplayer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1, stride=1, bias=False), # 3
            apply_normalization(64),
            nn.ReLU(),
        )
        # Layer1 -
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding=1, stride=1, bias=False), # 3
            nn.MaxPool2d(2, 2),
            apply_normalization(128),
            nn.ReLU(),
        )
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        self.reslayer1 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=1, stride=1, bias=False), # 3
            apply_normalization(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1, stride=1, bias=False), # 3
            apply_normalization(128),
            nn.ReLU(),
        )
        # Conv 3x3 [256k]
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=1, stride=1, bias=False), # 3
            nn.MaxPool2d(2, 2),
            apply_normalization(256),
            nn.ReLU(),
        )
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), padding=1, stride=1, bias=False), # 3
            nn.MaxPool2d(2, 2),
            apply_normalization(512),
            nn.ReLU(),
        )
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        self.reslayer2 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), padding=1, stride=1, bias=False), # 3
            apply_normalization(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1, stride=1, bias=False), # 3
            apply_normalization(512),
            nn.ReLU(),
        )
        self.maxpool3 = nn.MaxPool2d(4, 2)
        self.linear1 = nn.Linear(512,10)

        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self,x):
        x = self.preplayer(x)
        x1 = self.convlayer1(x)
        x2 = self.reslayer1(x1)
        x = x1+x2
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x1 = self.reslayer2(x)
        x = x+x1
        x = self.maxpool3(x)
        x = x.view(-1, 512)
        x = self.linear1(x)
        return F.log_softmax(x, dim=-1)


    def training_step(self, batch, batch_id):
        x, y = batch
        criterian=nn.CrossEntropyLoss()
        logits = self(x)
        loss = criterian(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        # self.log("train_loss", loss)

        # if stage:
        self.log(f"train_loss", loss, prog_bar=True)
        self.log(f"train_acc", self.train_accuracy, prog_bar=True)

        return loss

    def evaluate(self, batch,stage=None):
        x, y = batch
        criterian=nn.CrossEntropyLoss()
        logits = self(x)
        loss = criterian(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", self.val_accuracy, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
      optimizer =  torch.optim.Adam(self.parameters(), 
                          lr=self.learning_rate, 
                          weight_decay=self.weight_decay)
      scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.max_lr,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.epochs,
                pct_start=self.pct_start,
                div_factor=100,
                three_phase=False,
                final_div_factor=100,
                anneal_strategy='linear',verbose=False),
            "interval": "step",
        }
      return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}