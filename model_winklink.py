import torch.nn as nn
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from timm.layers import trunc_normal_
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch.nn.functional as F
from torch.fft import fft
import torchvision.utils as vutils
from einops import repeat
import lightning as L
from utils import pad, unpad, silog
from optimizer import get_optimizer
from metrics import compute_metrics
from utils import eigen_crop, garg_crop, custom_crop, no_crop
from dataset import DepthDataset
from torch.utils.data import DataLoader
import os

class PRN(nn.Module):
    def __init__(self, recurrent_iter=5, use_GPU=True):
        super(PRN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        n_channel = 3
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(2*n_channel, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, n_channel, 3, 1, 1),
        )

    def forward(self, input):

        x = input

        for _ in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input

        return x

class DestripeModel(L.LightningModule):

    def __init__(self, args, lr=8.31e-5, f_weight=0.001, stripe_weight=1e-4, recurrent_iter=4) -> None:
        super().__init__()
        self.learning_rate = lr
        self.f_weight = f_weight
        self.stripe_weight = stripe_weight

        self.args = args

        self.save_hyperparameters()

        self.stripe = PRN(recurrent_iter=recurrent_iter)

        self.register_buffer("mask", torch.zeros(1, args.crop_h, args.crop_w, dtype=torch.bool))
        self.mask[:, 0:3, :] = True
        self.mask[:, -3:, :] = True

    def forward(self, striped):
        # task 1: denoising
        destriped_predict = self.stripe(striped)
        # task 2: extract stripe
        # input2 = torch.concat([destriped_predict, striped], dim=1)
        # stripes_predict = self.model2(input2)
        
        return destriped_predict

    def loss_function(self, destriped_predict, destriped_gt):
         # loss 1: spatial domain loss
        loss_spatial = F.mse_loss(destriped_predict, destriped_gt)
        # loss 2: frequency domain loss
        freq_gt = torch.abs(fft(destriped_gt, dim=-1)).masked_fill_(self.mask, value=0)
        freq_predict = torch.abs(fft(destriped_predict, dim=-1)).masked_fill_(self.mask, value=0)
        loss_freq = F.mse_loss(freq_gt, freq_predict) * self.f_weight
        # loss 3: stripe reconstruction
        # loss_stripe = F.mse_loss(stripes_predict, stripes_gt) * self.stripe_weight

        return loss_spatial, loss_freq

    def training_step(self, batch, batch_idx):
        # load data
        striped, destriped_gt= batch['stripe'], batch['image']
        
        # model inference
        destriped_predict = self(striped)
        
        # loss integration
        loss_spatial, loss_freq = self.loss_function(destriped_predict, destriped_gt)
        loss = loss_spatial + loss_freq

        self.log_dict({"Loss": loss.detach().clone(), 
                        "Spatial Loss": loss_spatial.detach().clone(), 
                        "Freq Loss": loss_freq.detach().clone()},
                        sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # load data
        striped, destriped_gt = batch['stripe'], batch['image']
        
        # model inference
        destriped_predict = self(striped)
        
        # loss integration
        loss_spatial, loss_freq= self.loss_function(destriped_predict, destriped_gt)
        loss = loss_spatial + loss_freq

        self.log_dict({"Loss (val)": loss.detach().clone(), 
                        "Spatial Loss (val)": loss_spatial.detach().clone(), 
                        "Freq Loss (val)": loss_freq.detach().clone(),},
                        sync_dist=True)

    def on_validation_end(self) -> None:
        with torch.no_grad():
            show_dataloader = self.trainer.val_dataloaders
            test_striped, test_origin, destriped_predict = [], [], []
            num =16

            for idx, batch in enumerate(show_dataloader):
                if idx >= num:
                    break
                stripe, image = batch['stripe'], batch['image']
                test_origin.append(image)
                test_striped.append(stripe)
                image = image.to(self.device)
                stripe = stripe.to(self.device)

                destripe = self(stripe)
                destriped_predict.append(destripe)

            test_striped = torch.stack(test_striped[:num]).squeeze(1)
            test_origin  = torch.stack(test_origin[:num]).squeeze(1)
            destriped_predict = torch.stack(destriped_predict[:num]).squeeze(1)

            vutils.save_image(destriped_predict.data,
                            os.path.join(self.logger.log_dir, 
                                        "Reconstructions", 
                                        f"Epoch_{self.current_epoch}.png"),
                            normalize=True,
                            nrow=4)
            
            vutils.save_image(test_striped.data,
                            os.path.join(self.logger.log_dir, 
                                        "Striped", 
                                        f"striped.png"),
                            normalize=True,
                            nrow=4)
            
            vutils.save_image(test_origin.data,
                            os.path.join(self.logger.log_dir, 
                                        "Origin", 
                                        f"image.png"),
                            normalize=True,
                            nrow=4)

    def train_dataloader(self):
        train_dataset = DepthDataset(
            args=self.args, 
            is_train=True, 
            filenames_path=self.args.train_filenames_path, 
            data_path=self.args.train_data_path, 
            depth_factor=self.args.train_depth_factor
        )
        return DataLoader(train_dataset, num_workers=self.args.num_workers, batch_size=self.args.batch_size)
        
    def val_dataloader(self):
        val_dataset = DepthDataset(
            args=self.args, 
            is_train=False, 
            filenames_path=self.args.val_filenames_path, 
            data_path=self.args.val_data_path, 
            depth_factor=self.args.val_depth_factor
        )
        return DataLoader(val_dataset, num_workers=self.args.num_workers)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
