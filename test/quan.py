import sys
sys.path.append("..")
from dataset import DepthDataset, SUNRGBDDepthDataset, IBIMSDepthDataset, DIODEDepthDataset, HYPERDepthDataset
from lightning.pytorch.loggers import TensorBoardLogger
import json
from pathlib import Path
from torch.utils.data import DataLoader
from modelshow import EcoDepth
import lightning as L
import torch
from utils import download_model
class Args:
    def __init__(self):
        with open("test_config2.json", "r") as f:
            config = json.load(f) 
        for n, v in config.items():
           setattr(self, n, v)

args = Args()

import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="A simple example of argparse.")

# 添加参数
parser.add_argument("--dataset", type=str, required=False, default='none', help="Name of the user")
parser.add_argument("--kp", type=int, required=False, default=3, help="Name of the user")
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

tb_logger = TensorBoardLogger(save_dir="/mnt/DataCenter1/jinming_xu/hugespace/ecodepth/show_output", name="showing")
# Path(f"{tb_logger.log_dir}/image").mkdir(exist_ok=True, parents=True)
# Path(f"{tb_logger.log_dir}/gt").mkdir(exist_ok=True, parents=True)
# Path(f"{tb_logger.log_dir}/stripe").mkdir(exist_ok=True, parents=True)
# Path(f"{tb_logger.log_dir}/prediction").mkdir(exist_ok=True, parents=True)
# Path(f"{tb_logger.log_dir}/destriped").mkdir(exist_ok=True, parents=True)

test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)

trainer = L.Trainer(logger=tb_logger, use_distributed_sampler=False, devices=[args.gpu])

trainer.test(model, dataloaders=test_loader)


# def save_depth_with_colormap(depth_map, output_path):
#     # Create mask of valid (non-NaN) values
#     valid_mask = ~np.isnan(depth_map)

#     # Create an array for normalized depth, initialize with zeros
#     norm_depth = np.zeros_like(depth_map, dtype=np.float32)

#     # Normalize only valid pixels to [0, 255]
#     min_val = np.nanmin(depth_map)
#     max_val = np.nanmax(depth_map)
#     norm_depth[valid_mask] = (depth_map[valid_mask] - min_val) / (max_val - min_val) * 255

#     # Convert to uint8
#     depth_8u = np.uint8(norm_depth)

#     # Apply colormap
#     colored = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

#     # Set NaN regions to black
#     colored[~valid_mask] = [0, 0, 0]

#     # Save the image
#     cv2.imwrite(output_path, colored)