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

def make_trainer(max_epochs,train_loader,test_loader,refresh_rate=10,accelerator="auto",
            tensorboard_logs = "tf_logs/",
            csv_logs = "csv+logs/"
            ):
    tb_logger = pl_loggers.TensorBoardLogger(tensorboard_logs)
    csv_logger = CSVLogger(save_dir=csv_logs)

    model = CustomResnetModule()
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
    
def save_checkpoints(trainer):
  device = torch.device("cpu")
  trainer.save_checkpoint("best.ckpt")
  best_model = torch.load("best.ckpt")
  torch.save(best_model['state_dict'], f'best_model.pth')
  litemodel = CustomResnet()
  litemodel.load_state_dict(torch.load("best_model.pth",map_location='cpu'))
  device = "cpu"
  return litemodel,"best_model.pth"