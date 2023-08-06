from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from custom_models.dataset import CIFARDataModule
from custom_models.lightning_playground.modules.custom_resnet import CustomResnetModule
from pytorch_lightning.loggers import CSVLogger
from custom_models.custom_resnet import CustomResnet
import torch 
from pytorch_lightning import Trainer
import pandas as pd 
from IPython.core.display import display
import seaborn as sn
import os 



def make_trainer(max_epochs,train_loader,test_loader,max_lr,
                 learning_rate=0.01,weight_decay=1e-4,
                 refresh_rate=10,accelerator="auto",
                 tensorboard_logs = "tf_logs/",
                 csv_logs = "csv+logs/"
            ):
    tb_logger = pl_loggers.TensorBoardLogger(tensorboard_logs)
    csv_logger = CSVLogger(save_dir=csv_logs)

    model = CustomResnetModule(max_lr,
                               learning_rate,
                               weight_decay,
                               steps_per_epoch=len(train_loader),
                               pct_start=5/max_epochs,
                               epochs=max_epochs)
    data_module = CIFARDataModule(train_loader,test_loader)
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=[tb_logger, csv_logger],
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=refresh_rate)],
    )
    trainer.fit(model,data_module)
    return trainer

def evaluate_performace(csv_log_file_path):
    metrics = pd.read_csv(csv_log_file_path)
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    display(metrics.dropna(axis=1, how="all").head())
    sn.relplot(data=metrics, kind="line")
    
def save_checkpoints(path="/content/tf_logs/lightning_logs/"):
  versions = os.listdir(path)
  versionid = []
  for i in versions:
    versionid.append(int(i.replace("version_","")))

  best_weight_folder = os.path.join(path,f"version_{max(versionid)}","checkpoints")
  weights = os.listdir(best_weight_folder)[0]

  weights_path = os.path.join(best_weight_folder,weights)

  print(weights_path)
  device = torch.device("cpu")
  # trainer.save_checkpoint("best.ckpt")
  best_model = torch.load(weights_path)
  torch.save(best_model['state_dict'], f'best_model.pth')
  litemodel = CustomResnet()
  litemodel.load_state_dict(torch.load("best_model.pth",map_location='cpu'))
  device = "cpu"
  return litemodel,"best_model.pth"