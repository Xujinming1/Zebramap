import sys
sys.path.append("..")
import json
from model import Zebramap
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path
import torch


class Args:
    def __init__(self):
        with open("train_config.json", "r") as f:
            config = json.load(f) 
        for n, v in config.items():
           setattr(self, n, v)

def main():
    args = Args()

    model = Zebramap(args)

    model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu", weights_only=True)["state_dict"], strict=False)

    checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            monitor="val_loss",
            save_weights_only=True,
        )

    checkpoint_last = ModelCheckpoint(
            filename="{epoch}_end",
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
        devices=[args.gpu]
    )

    trainer.fit(
        model=model, 
    )


if __name__ == '__main__':
    main()