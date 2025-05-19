import sys
sys.path.append("..")
from dataset_origin import DepthDataset
import json
from torch.utils.data import DataLoader
from model_origin import EcoDepth
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path
import torch
from utils import download_model


class Args:
    def __init__(self):
        with open("train_config.json", "r") as f:
            config = json.load(f) 
        for n, v in config.items():
           setattr(self, n, v)

def main():
    args = Args()

    model = EcoDepth(args)

    if args.ckpt_path == "":
        model_str = f"weights_{args.scene}.ckpt"
        download_model(model_str)
        args.ckpt_path = f"../checkpoints/{model_str}"

    # model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu", weights_only=True)["state_dict"], strict=False)
    # model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu")["state_dict"], strict=False)

    train_dataset = DepthDataset(
        args=args, 
        is_train=True, 
        filenames_path=args.train_filenames_path, 
        data_path=args.train_data_path, 
        depth_factor=args.train_depth_factor
    )

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size)

    val_dataset = DepthDataset(
        args=args, 
        is_train=False, 
        filenames_path=args.val_filenames_path, 
        data_path=args.val_data_path, 
        depth_factor=args.val_depth_factor
    )

    val_loader = DataLoader(val_dataset, num_workers=args.num_workers)

    checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            monitor="val_loss",
            save_weights_only=True,
        )

    checkpoint_last = ModelCheckpoint(
            filename="{epoch}-{step}",
            save_top_k=1,  # 只保留一个最新的权重
            every_n_epochs=1  # 每个 epoch 末尾保存
        )

    checkpoint_d1 = ModelCheckpoint(
            filename="best-{d1:.6f}-{epoch}-{step}",
            monitor="d1",
            mode="max",
            save_top_k=1,
            save_weights_only=True
        )


    tb_logger = TensorBoardLogger(save_dir="/mnt/DataCenter1/jinming_xu/data_ecodepth/pl_output", name="training")
    Path(f"{tb_logger.log_dir}/image").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/gt").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/stripe").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/prediction").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/destriped").mkdir(exist_ok=True, parents=True)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        val_check_interval=args.val_check_interval,
        callbacks=[checkpoint_callback, checkpoint_last, checkpoint_d1],
        logger=tb_logger,
        # strategy="",
        devices=[0],
    )

    trainer.fit(
        model=model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        # ckpt_path=args.ckpt_path
    )


if __name__ == '__main__':
    main()