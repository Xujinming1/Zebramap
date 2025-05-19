import sys
sys.path.append("..")
from dataset import DepthDataset
import json
from torch.utils.data import DataLoader
from model_winklink import DestripeModel
import lightning as L
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

    model = DestripeModel(args)

    # model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu", weights_only=True)["state_dict"], strict=False)
    model.load_state_dict(torch.load(args.wink_path, map_location="cpu", weights_only=False)["state_dict"], strict=False)

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


    tb_logger = TensorBoardLogger(save_dir="/mnt/DataCenter1/jinming_xu/data_ecodepth/pl_output", name="winklink")
    Path(f"{tb_logger.log_dir}/Origin").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Striped").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        val_check_interval=args.val_check_interval,
        # callbacks=[checkpoint_callback, checkpoint_last, checkpoint_d1],
        logger=tb_logger,
        # strategy="",
        devices=[0]
    )

    trainer.fit(
        model=model, 
    )


if __name__ == '__main__':
    main()