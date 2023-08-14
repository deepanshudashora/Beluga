from lightning.pytorch.callbacks import ProgressBar
from lightning.pytorch import LightningModule, Trainer, seed_everything
import torch 
from torchmetrics import Accuracy
import torch.optim as optim
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
import pandas as pd 
from IPython.core.display import display


import pandas as pd 
from custom_models.YOLOv3.loss import YoloLoss
from IPython.core.display import display
import seaborn as sn
from custom_models.YOLOv3.model import YOLOv3
from custom_models.YOLOv3.dataset import YOLODataModule
import os 

def check_accuracy(x,y,model,threshold,device):        
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for i in range(3):
        y[i] = y[i].to(device)
        obj = y[i][..., 0] == 1 # in paper this is Iobj_i
        noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

        correct_class += torch.sum(
            torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
        )
        tot_class_preds += torch.sum(obj)

        obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
        correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
        tot_obj += torch.sum(obj)
        correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
        tot_noobj += torch.sum(noobj)
        


def check_accuracy(x,y,model,threshold,device):        
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for i in range(3):
        y[i] = y[i].to(device)
        obj = y[i][..., 0] == 1 # in paper this is Iobj_i
        noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

        correct_class += torch.sum(
            torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
        )
        tot_class_preds += torch.sum(obj)

        obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
        correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
        tot_obj += torch.sum(obj)
        correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
        tot_noobj += torch.sum(noobj)
        
class YOLOTraining(LightningModule):
    def __init__(self,loss_fn,config,model,max_lr,train_loader,pct_start):
        super().__init__()
        #self.train_accuracy = Accuracy(task="multiclass", num_classes=20)
        #self.val_accuracy = Accuracy(task="multiclass", num_classes=20)
        self.config = config
        self.loss_fn = loss_fn
#         self.scaled_anchors = (torch.tensor(self.config.ANCHORS)
#                                 * torch.tensor(self.config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
#                             )
        self.ANCHORS = self.config.ANCHORS
        self.S = self.config.S
        self.model = model
        self.max_lr = max_lr
        self.pct_start = pct_start
        self.train_loader = train_loader
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_id):
        x,y=batch
        x = x
        y0, y1, y2 = (
            y[0],
            y[1],
            y[2],
        )
        scaled_anchors = (
            torch.tensor(self.ANCHORS)
            * torch.tensor(self.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(y0.device)
        with torch.cuda.amp.autocast():
            out = self(x)
            loss = (
                self.loss_fn(out[0], y0, scaled_anchors[0])
                + self.loss_fn(out[1], y1, scaled_anchors[1])
                + self.loss_fn(out[2], y2, scaled_anchors[2])
            )

            self.log(f"train_loss", loss, prog_bar=True,sync_dist=True)

        return loss
    
    def evaluate(self, batch,stage=None):
        x,y=batch
        x = x
        y0, y1, y2 = (
            y[0],
            y[1],
            y[2],
        )
        scaled_anchors = (
            torch.tensor(self.ANCHORS)
            * torch.tensor(self.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(y0.device)
        with torch.cuda.amp.autocast():
            out = self(x)
            loss = (
                self.loss_fn(out[0], y0, scaled_anchors[0])
                + self.loss_fn(out[1], y1, scaled_anchors[1])
                + self.loss_fn(out[2], y2, scaled_anchors[2])
            )
            #             val_accuracy = check_accuracy(x,y,self.model,self.config.CONF_THRESHOLD,"cuda")
            #             if val_accuracy!=None:
            #                 clas_acc,no_obj_acc,obj_acc = val_accuracy[0],val_accuracy[1],val_accuracy[2]
            #                 self.log(f"{stage}_loss", loss, prog_bar=True,sync_dist=True)
            #                 self.log(f"{stage}_clasacc", clas_acc, prog_bar=True,sync_dist=True)
            #                 self.log(f"{stage}_no_obj_acc", no_obj_acc, prog_bar=True,sync_dist=True)
            #                 self.log(f"{stage}_obj_acc", obj_acc, prog_bar=True,sync_dist=True)
            #             else:
            self.log(f"{stage}_loss", loss, prog_bar=True,sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")
        
    def on_epoch_end(self):
        # Get train and validation losses from the trainer and epoch_logs
        train_loss = self.trainer.callback_metrics["train_loss"]
        val_loss = self.trainer.callback_metrics["val_loss"]

        # Print train and validation losses
        self.log(f"Epoch {self.current_epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}",prog_bar=True,sync_dist=True)

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
      optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY
        )
      scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.max_lr,
                steps_per_epoch=len(self.train_loader),
                epochs=self.config.NUM_EPOCHS,
                pct_start=self.pct_start,
                div_factor=100,
                three_phase=False,
                final_div_factor=100,
                anneal_strategy='linear',verbose=False),
      }
      return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
  
def make_trainer(max_epochs,train_loader, test_loader, train_eval_loader,max_lr,
                 learning_rate,weight_decay,check_val_every_n_epoch,config,precision=16,
                 refresh_rate=10,accelerator="auto",
                 tensorboard_logs = "tf_logs/",
                 csv_logs = "csv_training_logs/"
            ):
    tb_logger = pl_loggers.TensorBoardLogger(tensorboard_logs)
    csv_logger = CSVLogger(save_dir=csv_logs)    
    loss_fn = YoloLoss()
    model = YOLOv3(num_classes=config.NUM_CLASSES)
    model = YOLOTraining(loss_fn,config,model,max_lr,train_loader,5/max_epochs)
    data_module = YOLODataModule(train_loader, test_loader, train_eval_loader)
    from lightning.pytorch.strategies import DDPStrategy 
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu", 
        devices=2, 
        strategy="ddp_notebook_find_unused_parameters_true",
        logger=[tb_logger, csv_logger],
        precision=16,
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
        check_val_every_n_epoch=check_val_every_n_epoch
    )
    trainer.fit(model,data_module)
    return trainer