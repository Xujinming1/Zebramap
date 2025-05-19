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
import numpy as np
import matplotlib
import cv2

NUM_DECONV = 3
NUM_FILTERS = [32, 32, 32]
DECONV_KERNELS = [2, 2, 2]
# VIT_MODEL = 'google/vit-base-patch16-224'
VIT_MODEL = '/mnt/DataCenter1/jinming_xu/data_ecodepth/vit_model'


def pad_to_make_square(x):
    y = 255*((x+1)/2)
    y = torch.permute(y, (0,2,3,1))
    bs, _, h, w = x.shape
    if w>h:
        patch = torch.zeros(bs, w-h, w, 3).to(x.device)
        y = torch.cat([y, patch], axis=1)
    else:
        patch = torch.zeros(bs, h, h-w, 3).to(x.device)
        y = torch.cat([y, patch], axis=2)
    return y.to(torch.int)

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

class EmbeddingAdapter(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, texts, gamma):
        emb_transformed = self.fc(texts)
        texts = texts + gamma * emb_transformed
        texts = repeat(texts, 'n c -> n b c', b=1)
        return texts

class EcoDepthEncoder(nn.Module):
    def __init__(
        self, 
        out_dim=1024, 
        ldm_prior=[320, 640, 1280+1280], 
        sd_path=None, 
        emb_dim=768, 
        args=None,
        train_from_scratch=False,
    ):
        super().__init__()

        self.args = args

        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )
        
        if train_from_scratch:
            self.apply(self._init_weights)
        
        if args.method in [5, 6, 7, 9, 10, 11]:
            self.cide_module = CIDE(args, emb_dim//2, train_from_scratch)
            if args.indenco:
                self.cide_module2 = CIDE(args, emb_dim//2, train_from_scratch)
        else:
            self.cide_module = CIDE(args, emb_dim, train_from_scratch)
        
        self.config = OmegaConf.load('../v1-inference.yaml')
        if self.args.method == 3:
            unet_config = self.config.model.params.unet_config2
        else:
            unet_config = self.config.model.params.unet_config

        first_stage_config = self.config.model.params.first_stage_config
        
        if train_from_scratch:
            if sd_path is None:
                sd_path = '../../checkpoints/v1-5-pruned-emaonly.ckpt'
            # unet_config.params.ckpt_path = sd_path
        
        self.unet = instantiate_from_config(unet_config)
        self.encoder_vq = instantiate_from_config(first_stage_config)
        del self.encoder_vq.decoder
        del self.unet.out

        for param in self.encoder_vq.parameters():
            param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, stripe=None):
        with torch.no_grad():
            # convert the input image to latent space and scale.
            latents = self.encoder_vq.encode(x).mode().detach() * self.config.model.params.scale_factor

        conditioning_scene_embedding = self.cide_module(x)

        # print(x)
        # print(stripe)
        # exit(0)

        if stripe is not None:            
            # print('11')
            # exit(0)
            if self.args.method in [6, 7, 9, 10, 11]:
                input = stripe
            else:
                input = stripe-x
                input[input < -1] = -1
                input[input > 1] = 1

            if self.args.indenco:
                conditioning_scene_embedding2 = self.cide_module2(input)
            else:
                conditioning_scene_embedding2 = self.cide_module(input)

            if self.args.method in [3, 5, 6, 7, 9, 10, 11]:
                conditioning_scene_embedding = torch.concat((conditioning_scene_embedding, conditioning_scene_embedding2), dim=2)
            elif self.args.method == 4:
                conditioning_scene_embedding = conditioning_scene_embedding2

        t = torch.ones((x.shape[0],), device=x.device).long()
        outs = self.unet(latents, t, context=conditioning_scene_embedding)

        feats = [outs[0], outs[1], torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1)]
        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        return self.out_layer(x)

class CIDE(nn.Module):
    def __init__(self, args, emb_dim, train_from_scratch):
        super().__init__()
        self.args = args
        self.vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL, resume_download=True)
        # if train_from_scratch:
        #     vit_config = ViTConfig(num_labels=1000)
        #     self.vit_model = ViTForImageClassification(vit_config)
        # else:
        self.vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL, resume_download=True)

        for param in self.vit_model.parameters():
            param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(1000, 400),
            nn.GELU(),
            nn.Linear(400, args.no_of_classes)
        )
        self.dim = emb_dim
        self.m = nn.Softmax(dim=1)
        
        self.embeddings = nn.Parameter(torch.randn(self.args.no_of_classes, self.dim))
        self.embedding_adapter = EmbeddingAdapter(emb_dim=self.dim)
        
        self.gamma = nn.Parameter(torch.ones(self.dim) * 1e-4)
    
    def forward(self, x):
        y = pad_to_make_square(x)
        # use torch.no_grad() to prevent gradient flow through the ViT since it is kept frozen
        with torch.no_grad():
            inputs = self.vit_processor(images=y, return_tensors="pt").to(x.device)
            vit_outputs = self.vit_model(**inputs)
            vit_logits = vit_outputs.logits
            
        class_probs = self.fc(vit_logits)
        class_probs = self.m(class_probs)
        
        class_embeddings = class_probs @ self.embeddings
        conditioning_scene_embedding = self.embedding_adapter(class_embeddings, self.gamma) 
        
        return conditioning_scene_embedding
        
class EcoDepth(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.max_depth = args.max_depth

        self.args = args
        embed_dim = 192
        channels_in = embed_dim * 8
        channels_out = embed_dim

        self.save_hyperparameters()

        if self.args.method == 1:
            self.fuze = nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1)
        elif self. args.method == 2:
            self.fuze = nn.Conv2d(embed_dim*16, embed_dim*8, 3, 1, 1)
            
        self.stripe = PRN(recurrent_iter=4)

        if args.winkfreeze:
            for param in self.stripe.parameters():
                param.requires_grad = False

        self.encoder = EcoDepthEncoder(out_dim=channels_in, args = args, train_from_scratch=args.train_from_scratch)
        self.decoder = Decoder(channels_in, channels_out, args)

        self.light_vector = (0, 0, 1)
        
        if args.eval_crop == "eigen":
            self.eval_crop = eigen_crop
        elif args.eval_crop == "garg":
            self.eval_crop = garg_crop
        elif args.eval_crop == "custom":
            self.eval_crop = custom_crop
        else:
            self.eval_crop = no_crop
        
        # Only support finetuning for now
        # assert not args.train_from_scratch
        
        if args.train_from_scratch:
            self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
            
        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.register_buffer("mask", torch.zeros(1, args.crop_h, args.crop_w, dtype=torch.bool))
        self.mask[:, 0:3, :] = True
        self.mask[:, -3:, :] = True

    def forward(self, x, s=None, normal=None):
        # x must be a pytorch tensor of shape (bs, 3, h, w)
        # and the each value ranges between [0, 1]
        _, _, h, _ = x.shape

        kp = self.args.kp
        # kp = 20

        if s is not None:
            if not self.args.Onorm_module:
                with torch.no_grad():
                    x = self.stripe(s)
                if self.args.method in [6, 7, 9, 10, 11]:
                    # print(s)
                    s = (s - x) / kp / torch.float_power(torch.abs(normal[:, 2, :, :]).unsqueeze(1), 0.45)
                    # print(x)
                    # print(torch.float_power(torch.abs(normal[:, 2, :, :]).unsqueeze(1), 0.45))
                    # print(s)
                    # exit(0)
                    # vutils.save_image(s.data,
                    #     os.path.join(self.logger.log_dir, 
                    #                 "stripe", 
                    #                 f"stripe.png"),
                    #     normalize=False,
                    #     nrow=4)
                    s = torch.clamp(s, min=0, max=1.0)
            if self.args.method == 11: 
                s = s*2.0 - 1.0
                s, padding = pad(s, 64)

        x_destriped = x
        x = torch.clamp(x, min=0, max=1.0)
        x = x*2.0 - 1.0  # normalize to [-1, 1]
        x, padding = pad(x, 64)

        if self.args.method in [3, 4, 5, 6, 7, 9, 10, 11]:
            conv_feats = self.encoder(x, s)
        else:
            conv_feats = self.encoder(x)

        if self.args.method == 2:
            conv_feats2 = self.encoder(s)
            conv_feats = self.fuze(torch.cat((conv_feats, conv_feats2), dim=1))

        out = self.decoder([conv_feats])
        out = unpad(out, padding)

        if s is not None and self.args.method == 1:
            conv_feats2 = self.encoder(s)
            out2 = self.decoder([conv_feats2])
            out2 = unpad(out2, padding)

            out = self.fuze(torch.cat((out, out2), dim=1))

        out_depth = self.last_layer_depth(out)            
        pred = torch.sigmoid(out_depth) * self.max_depth
        # pred is a pt of shape (bs, 1, h, w)
        # where each value ranges between [0, self.max_depth]
        return pred, x_destriped
    
    def mse(self, pred, gt):
        if self.args.use_stripe and not self.args.winkfreeze:
            loss_spatial = F.mse_loss(pred, gt)

            dim = -2
            freq_gt = torch.abs(fft(gt, dim=dim)).masked_fill_(self.mask, value=0)
            freq_predict = torch.abs(fft(pred, dim=dim)).masked_fill_(self.mask, value=0)
            loss_freq = F.mse_loss(freq_gt, freq_predict) * 1.0

            return loss_freq + loss_spatial * 1000
        
        else:
            return 0

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
    
    def training_step(self, batch, batch_idx):
        image, depth = batch["image"], batch["depth"]

        if self.args.use_stripe:
            stripe = batch["stripe"]
        else:
            stripe = None

        pred, destriped = self(image, stripe)
        loss = silog(pred, depth) + self.mse(destriped, image)

        self.log(f"train_loss", loss)
        return loss
    
    def _shared_eval_step(self, batch, batch_idx, prefix):
        with torch.no_grad():
            image, depth = batch["image"], batch["depth"]

            if self.args.use_stripe:
                stripe = batch["stripe"]
                stripe_concat = torch.cat([stripe, stripe.flip(-1)])
                if self.args.realsense and self.args.method >= 7:
                    normal = batch['normal']
                    normal_concat = torch.cat([normal, normal.flip(-1)])
                else:
                    normal = None
                    normal_concat = None
            else:
                stripe = None
                stripe_concat = None
                normal = None
                normal_concat = None

            image_concat = torch.cat([image, image.flip(-1)])
            pred_concat, destriped = self(image_concat, stripe_concat, normal_concat)
            pred = ((pred_concat[0] + pred_concat[1].flip(-1))/2).unsqueeze(0)

            if self.args.use_stripe:
                destriped = ((destriped[0] + destriped[1].flip(-1))/2).unsqueeze(0)
            
            if depth.shape[-2:] != pred.shape[-2:] :
                pred = torch.nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            depth = self.eval_crop(depth)

            loss = silog(pred, depth) + self.mse(destriped, image)

            metrics = compute_metrics(pred, depth, self.args)

            if metrics is None:
                return None

            self.log(f"{prefix}_loss", loss, sync_dist=True)
            self.log_dict(metrics)

            if 'd1' in metrics.keys():
                if np.isnan(metrics['d1'].item()) and prefix == 'test':
                    print('ji')
                    exit(0)

            torch.cuda.empty_cache()
            return None

    def saveoutputs(self, dataloader):
        with torch.no_grad():
            show_dataloader = dataloader
            images, stripes, depths, preds, destripes = [], [], [], [], []
            num = 100

            for idx, batch in enumerate(show_dataloader):
                if idx >= num:
                    break
                image, depth = batch['image'], batch['depth']
                images.append(image)
                depths.append(depth)
                image = image.to(self.device)
                # depth = depth.to(self.device)

                if self.args.use_stripe:
                    stripe = batch['stripe']
                    stripes.append(stripe)
                    stripe = stripe.to(self.device)
                    if self.args.realsense and self.args.method >= 7:
                        normal = batch['normal']
                        normal = normal.to(self.device)
                    else:
                        normal = None
                else:
                    stripe = None
                    normal = None

                pred, destripe = self(image, stripe, normal)

                preds.append(pred)
                
                if self.args.use_stripe:
                    destripes.append(destripe)

                # cmap = matplotlib.colormaps.get_cmap('Spectral_r')
                # # gt_norm = gt_norm.squeeze(0).squeeze(0)
                # # print(shape)
                # # print(pred[0,0].shape)
                # # depth = depth[45:471, 41:601]
                # print(depth.device)
                # print(pred.device)
                # if depth.shape[-2:] != pred.shape[-2:] :
                #     pred = torch.nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
                # depth = self.eval_crop(depth)
                # metrics = compute_metrics(pred, depth, self.args)
                # MIN_DEPTH_EVAL = 1e-3
                # pred[pred > self.args.max_depth] = self.args.max_depth
                # pred[pred < MIN_DEPTH_EVAL] = MIN_DEPTH_EVAL
                # valid_mask = torch.logical_and(depth > MIN_DEPTH_EVAL, depth < self.args.max_depth)
                # evalmask = torch.zeros(depth.shape[-2:]).to(self.device)
                # evalmask[45:471, 41:601] = 1
                # valid_mask = torch.logical_and(valid_mask, evalmask)
                # depth = (depth - depth[valid_mask].min()) / (depth[valid_mask].max() - depth[valid_mask].min()) * 255
                # depth = depth[0, 0].detach().cpu().numpy().astype(np.uint8)
                # depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                # cv2.imwrite(os.path.join(self.logger.log_dir, 
                #                     "gt", 
                #                     f"gt{idx}-{metrics['d1']}-{metrics['rmse']}.png"), depth[45:471, 41:601])
                # # print(pred.shape)
                # pred = (pred- pred[valid_mask].min()) / (pred[valid_mask].max() - pred[valid_mask].min()) * 255
                # pred = pred[0, 0].detach().cpu().numpy().astype(np.uint8)
                # # pred = cv2.resize(pred, shape)
                # pred = (cmap(pred)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                # cv2.imwrite(os.path.join(self.logger.log_dir, 
                #                     "prediction", 
                #                     f"pred{idx}-{metrics['d1']}-{metrics['rmse']}.png"), pred[45:471, 41:601])
                    

            images = torch.stack(images[:num]).squeeze(1)
            depths = torch.stack(depths[:num]).squeeze(1)
            preds = torch.stack(preds[:num]).squeeze(1)
            if self.args.use_stripe:
                stripes = torch.stack(stripes[:num]).squeeze(1)
                destripes = torch.stack(destripes[:num]).squeeze(1)
            else:
                stripes = None

            vutils.save_image(images.data,
                        os.path.join(self.logger.log_dir, 
                                    "image", 
                                    f"image.png"),
                        normalize=True,
                        nrow=4)
        
            gt_norm = depths
            N, H, W = gt_norm.shape[0], gt_norm.shape[2], gt_norm.shape[3]

            vutils.save_image(gt_norm.view(N,1,H,W).data,
                            os.path.join(self.logger.log_dir, 
                                        "gt", 
                                        f"gt.png"),
                            normalize=True,
                            nrow=4)
            
            vutils.save_image(preds.data,
                            os.path.join(self.logger.log_dir, 
                                        "prediction", 
                                        f"depth_pred{self.global_step}.png"),
                            normalize=True,
                            nrow=4)
            
            if self.args.use_stripe:
                vutils.save_image(stripes.data,
                            os.path.join(self.logger.log_dir, 
                                        "stripe", 
                                        f"stripe.png"),
                            normalize=False,
                            nrow=4)
                
                vutils.save_image(destripes.data,
                                os.path.join(self.logger.log_dir, 
                                            "destriped", 
                                            f"destriped{self.global_step}.png"),
                                normalize=True,
                                nrow=4)
                

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val")

    def on_validation_end(self):
        return self.saveoutputs(self.trainer.val_dataloaders)
    
    def test_step(self, batch, batch_idx):
        # cmap = matplotlib.colormaps.get_cmap('coolwarm')

        # depth = batch['depth']
        # depth = (depth.cpu().numpy().squeeze(0).squeeze(0)*10000).astype(np.uint16)
        # cv2.imwrite(os.path.join(self.logger.log_dir, f'd{batch_idx}.png'), depth)
        # stripe = batch['stripe']
        # stripe = stripe.cpu().numpy().squeeze(0).transpose(1, 2, 0)*255.0
        # # cv2.cvtColor(stripe, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(self.logger.log_dir, f'{batch_idx}.png'), stripe)

        # return 

        cmap = matplotlib.colormaps.get_cmap('Spectral')
        # num = 100
        # if batch_idx < num:
        maxd = 10
        if 'hypersim' in self.args.test_data_path:
            maxd = 80
        elif 'sunrgbd' in self.args.test_data_path:
            maxd = 8
        elif 'realsense' in self.args.test_data_path:
            maxd = 10

        image, depth = batch['image'], batch['depth']
        if self.args.use_stripe:
            stripe = batch['stripe']
            if self.args.realsense and self.args.method >= 7:
                normal = batch['normal']
            else:
                normal = None
        else:
            stripe = None
            normal = None
        
        pred, destripe = self(image, stripe, normal)

        if depth.shape[-2:] != pred.shape[-2:] :
            pred = torch.nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
            image = torch.nn.functional.interpolate(image, depth.shape[-2:], mode='bilinear', align_corners=True)

        stripe = stripe.cpu().numpy().squeeze(0).transpose(1, 2, 0)*255.0
        image = image.cpu().numpy().squeeze(0).transpose(1, 2, 0)*255.0
        destripe = destripe.cpu().numpy().squeeze(0).transpose(1, 2, 0)*255.0
        depth = depth.cpu().numpy().squeeze(0).squeeze(0)
        pred = pred.cpu().numpy().squeeze(0).squeeze(0)
        # print(pred.shape)
        # maxd = 10
        mask = np.logical_and(depth > 1e-3, depth < maxd, ~np.isnan(depth))

        # absr = np.zeros_like(depth, np.float32)
        # absr[mask] = np.abs(depth[mask]-pred[mask])/depth[mask]
        
        # maxabs = np.max(absr)

        # absr = (cmap(absr)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # cv2.imwrite(os.path.join(self.logger.log_dir, f'{batch_idx}_{maxabs}.png'), np.concatenate((image, absr), axis=1))



        normdepth = np.zeros_like(depth, np.float32)
        normpred = np.zeros_like(depth, np.float32)
        # torch.isnan(depth)
        # torch.isinf(depth)
        vmax = np.max(depth[mask])
        vmin = np.min(depth[mask])

        normdepth[mask] = (depth[mask] - vmin) / (vmax - vmin)
        normpred = (pred - vmin) / (vmax - vmin)

        # normdepth = np.uint8(normdepth)
        # normpred = np.uint8(normpred)

        # cmap = matplotlib.colormaps.get_cmap('Spectral')
        # print(cmap(normdepth[mask]).shape)
        colordepth = (cmap(normdepth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        colorpred = (cmap(normpred)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


        # colordepth = cv2.applyColorMap(normdepth, cv2.COLORMAP_JET)
        # colorpred = cv2.applyColorMap(normpred, cv2.COLORMAP_JET)

        # colordepth = 

        # print(colordepth.shape)

        colordepth[~mask] = [0,0,0]
        # colordepth = colordepth[45: 471, 41:601]
        


        cv2.imwrite(os.path.join(self.logger.log_dir, f'{batch_idx}.png'), np.concatenate((stripe, destripe, image, colordepth, colorpred), axis=0))

            # exit(0)


        # return self._shared_eval_step(batch, batch_idx, "test")
        return
    
    def on_test_end(self):
        # measure_name = ['silog', 'abs_rel', 'log10', 'rmse', 'sq_rel', 'rmse_log', 'd1', 'd2', 'd3']
        # metrics = self.trainer.logged_metrics
        # outputpath = os.path.join(self.logger.log_dir, 'measures.txt')
        # with open(outputpath, 'w') as f:
        #     f.write("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}\n".format('silog', 'abs_rel', 'log10', 'rms',
        #                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
        #                                                                     'd3'))
        #     for i in range(8):
        #         f.write('{:7.4f}, '.format(metrics[measure_name[i]].item()))
        #     f.write('{:7.4f}\n'.format(metrics[measure_name[8]].item()))
        # return self.saveoutputs(self.trainer.test_dataloaders)
        return 
        
    def configure_optimizers(self):
        optimizer = get_optimizer(self, self.args)
        return optimizer

        
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = NUM_DECONV
        self.in_channels = in_channels
        self.args = args    
        self.deconv_layers = self._make_deconv_layer(
            NUM_DECONV,
            NUM_FILTERS,
            DECONV_KERNELS,
        )
    
        conv_layers = []
        conv_layers.append(
            nn.Conv2d(
                in_channels=NUM_FILTERS[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
        out = self.deconv_layers(conv_feats[0])
        out = self.conv_layers(out)

        out = self.up(out)
        out = self.up(out)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


