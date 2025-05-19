import sys
sys.path.append("..")
from lightning.pytorch.loggers import TensorBoardLogger
import json
from pathlib import Path
from model import Zebramap
import lightning as L
import torch
class Args:
    def __init__(self):
        with open("test_config.json", "r") as f:
            config = json.load(f) 
        for n, v in config.items():
           setattr(self, n, v)

args = Args()

model = Zebramap(args)

model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu", weights_only=False)["state_dict"], strict=False)

model.load_state_dict(torch.load(args.wink_path, map_location="cpu", weights_only=False)["state_dict"], strict=False)

tb_logger = TensorBoardLogger(save_dir="/mnt/DataCenter1/jinming_xu/hugespace/ecodepth/test_output", name="testing")
Path(f"{tb_logger.log_dir}/image").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/gt").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/stripe").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/prediction").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/destriped").mkdir(exist_ok=True, parents=True)


trainer = L.Trainer(logger=tb_logger, use_distributed_sampler=False, devices=[args.gpu], enable_progress_bar=True)

trainer.test(model)
