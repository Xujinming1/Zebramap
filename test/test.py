import sys
sys.path.append("..")
from dataset import DepthDataset, SUNRGBDDepthDataset, IBIMSDepthDataset, DIODEDepthDataset, HYPERDepthDataset
from lightning.pytorch.loggers import TensorBoardLogger
import json
from pathlib import Path
from torch.utils.data import DataLoader
from model import EcoDepth
import lightning as L
import torch
from utils import download_model
class Args:
    def __init__(self):
        with open("test_config.json", "r") as f:
            config = json.load(f) 
        for n, v in config.items():
           setattr(self, n, v)

args = Args()

import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="A simple example of argparse.")

# 添加参数
parser.add_argument("--dataset", type=str, required=False, default='none', help="Name of the user")
parser.add_argument("--kp", type=float, required=False, default=3, help="Name of the user")
arg = parser.parse_args()

if arg.dataset != 'none':
    args.test_filenames_path = arg.dataset
    args.kp = arg.kp

model = EcoDepth(args)

if args.ckpt_path == "":
    model_str = f"weights_{args.scene}.ckpt"
    download_model(model_str)
    args.ckpt_path = f"../checkpoints/{model_str}"

model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu", weights_only=False)["state_dict"], strict=False)

model.load_state_dict(torch.load(args.wink_path, map_location="cpu", weights_only=False)["state_dict"], strict=False)


testname = args.test_data_path

if "sunrgbd" in testname:
    test_dataset = SUNRGBDDepthDataset(
        args=args, 
        is_train=False, 
        filenames_path=args.test_filenames_path, 
        data_path=args.test_data_path, 
        depth_factor=args.test_depth_factor
    )
    
elif "ibims" in testname:
    test_dataset = IBIMSDepthDataset(
        args=args, 
        is_train=False, 
        filenames_path=args.test_filenames_path, 
        data_path=args.test_data_path, 
        depth_factor=args.test_depth_factor
    )

elif "diode" in testname:
    test_dataset = DIODEDepthDataset(
        args=args, 
        is_train=False, 
        filenames_path=args.test_filenames_path, 
        data_path=args.test_data_path, 
        depth_factor=args.test_depth_factor
    )

elif "hypersim" in testname:
    test_dataset = HYPERDepthDataset(
        args=args, 
        is_train=False, 
        filenames_path=args.test_filenames_path, 
        data_path=args.test_data_path, 
        depth_factor=args.test_depth_factor
    )

else:
    test_dataset = DepthDataset(
        args=args, 
        is_train=False, 
        filenames_path=args.test_filenames_path, 
        data_path=args.test_data_path, 
        depth_factor=args.test_depth_factor
    )

tb_logger = TensorBoardLogger(save_dir="/mnt/DataCenter1/jinming_xu/hugespace/ecodepth/test_output", name="testing")
Path(f"{tb_logger.log_dir}/image").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/gt").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/stripe").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/prediction").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/destriped").mkdir(exist_ok=True, parents=True)

test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)

trainer = L.Trainer(logger=tb_logger, use_distributed_sampler=False, devices=[args.gpu], enable_progress_bar=True)

trainer.test(model, dataloaders=test_loader)

############# for kp selection #############

print("kp ", args.kp)
# After test completes, get desired metrics
metrics = trainer.logged_metrics
d1 = metrics.get("d1", None)
abs_rel = metrics.get("abs_rel", None)
rmse = metrics.get("rmse", None)  # 'rmse' is used for 'rms' in your print format

# Print the values if they exist
if d1 is not None and abs_rel is not None and rmse is not None:
    print(f"d1: {d1:.4f}, abs_rel: {abs_rel:.4f}, rmse: {rmse:.4f}")
else:
    print("One or more requested metrics not found in logged_metrics.")