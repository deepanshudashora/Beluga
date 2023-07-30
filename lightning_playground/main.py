from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from custom_models.lightning_playground.dataset import CIFARDataModule
from custom_models.lightning_playground.models.custom_resnet import CustomResnetModule
from pytorch_lightning.loggers import CSVLogger
import torch 
from pytorch_lightning import Trainer
import pandas as pd 
from IPython.core.display import display
import seaborn as sn

def make_trainer():
    tb_logger = pl_loggers.TensorBoardLogger('tf_logs/')
    csv_logger = CSVLogger(save_dir="csv_logs/")

    model = CustomResnetModule()
    data_module = CIFARDataModule()
    trainer = Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=[tb_logger, csv_logger],
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )
    
    return trainer

def evaluate_performace(csv_log_file_path):
    metrics = pd.read_csv(csv_log_file_path)
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    display(metrics.dropna(axis=1, how="all").head())
    sn.relplot(data=metrics, kind="line")