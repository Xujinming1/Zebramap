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
from model_winklink import DestripeModel
import numpy as np
import os

class Args:
    def __init__(self):
        with open("./test/test_config.json", "r") as f:
            config = json.load(f) 
        for n, v in config.items():
           setattr(self, n, v)

args = Args()

model = DestripeModel(args)

if args.ckpt_path == "":
    model_str = f"weights_{args.scene}.ckpt"
    download_model(model_str)
    args.ckpt_path = f"../checkpoints/{model_str}"

model.load_state_dict(torch.load(args.wink_path, map_location="cpu", weights_only=False)["state_dict"], strict=False)
model.to(f"cuda:{args.gpu}")

print(f"Model is on: {next(model.parameters()).device}")

test_dataset = DepthDataset(
    args=args, 
    is_train=False, 
    filenames_path=args.test_filenames_path.replace('../', './'), 
    data_path=args.test_data_path, 
    depth_factor=args.test_depth_factor
)

# tb_logger = TensorBoardLogger(save_dir="/mnt/DataCenter1/jinming_xu/hugespace/ecodepth/test_output", name="testing")
# Path(f"{tb_logger.log_dir}/image").mkdir(exist_ok=True, parents=True)
# Path(f"{tb_logger.log_dir}/gt").mkdir(exist_ok=True, parents=True)
# Path(f"{tb_logger.log_dir}/stripe").mkdir(exist_ok=True, parents=True)
# Path(f"{tb_logger.log_dir}/prediction").mkdir(exist_ok=True, parents=True)
# Path(f"{tb_logger.log_dir}/destriped").mkdir(exist_ok=True, parents=True)

test_loader = DataLoader(test_dataset, num_workers=args.num_workers)

mindepth = 0.5
model.eval()
device = f'cuda:{args.gpu}'

with open('./kp.txt', 'a') as f:
    f.write('\n'+os.path.basename(args.test_filenames_path)+'\n')

for idx, batch in enumerate(test_loader):
    striped, depth, normal, onorm = batch['stripe'], batch['depth'].cpu().numpy().squeeze().squeeze(), batch['normal'].cpu().numpy().squeeze(), batch['image'].cpu().numpy().squeeze()
    striped = striped.to(device)

    image = model.stripe(striped).detach().cpu().numpy().squeeze()
    striped = striped.detach().cpu().numpy().squeeze()

    light_position = np.array([0.0, 0.0, 0.0])
    # 创建像素坐标的网格
    height, width = image.shape[-2:]
    # print(height, width)
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    fx = 585.0  # Focal length in pixels (horizontal)
    fy = 585.0  # Focal length in pixels (vertical)
    cx = 324.5  # Principal point (horizontal)
    cy = 253.7  # Principal point (vertical)

    # 将像素坐标转换为相机坐标 (X, Y, Z)
    depth[np.isnan(depth)] = mindepth-0.1
    depth[np.isinf(depth)] = 100
    depth[depth==0] = mindepth
    Z = depth
    X = (x_coords - cx) * Z / fx
    Y = (y_coords - cy) * Z / fy
    world_coords = np.dstack((X, Y, Z))  # 形状: (height, width, 3)

    # 计算光线向量（从光源到每个像素点）
    light_vectors = world_coords - light_position
    light_distances = np.linalg.norm(light_vectors, axis=2, keepdims=True)
    light_vectors_normalized = light_vectors / light_distances  # 归一化光线向量

    epsi = 0.05
    ratio = np.power(np.maximum(epsi, np.abs(np.sum(normal.transpose(1, 2, 0) * light_vectors_normalized, axis=2, keepdims=True))).transpose(2, 0, 1), 0.45)
    

    # print(depth)
    depth[np.isnan(depth)] = mindepth-0.1
    depth[np.isinf(depth)] = 100
    mask = np.logical_and(ratio > epsi, (depth > mindepth)[np.newaxis, :, :])
    maskk = np.repeat(mask, 3, axis=0)
    # mask = ratio > epsi
    # print(ratio)
    # print(mask)
    # print(onorm)
    # print((np.linalg.norm((ratio*onorm)[maskk]))**2)
    # print(striped-image)
    kp = np.inner(np.multiply(ratio, onorm)[maskk].flatten(), (striped-image)[maskk].flatten()) / (np.linalg.norm((ratio*onorm)[maskk]))**2
    # print(kp)
    # exit(0)

    with open('./kp.txt', 'a') as f:
        f.write(str(kp)+'\n')



    