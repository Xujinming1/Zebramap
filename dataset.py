import random
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import cv2
import os
from scipy import io
from syn4 import script_fsk_depth, script_fsk_depth_k0kp
import h5py

def readTXT(txt_path):
    with open(txt_path, 'r') as f:
        listInTXT = [line.strip() for line in f if "None" not in line]
    return listInTXT

class DepthDataset(Dataset):
    def __init__(self, args, is_train, filenames_path, data_path, depth_factor):
        super().__init__()
        self.is_train = is_train
        self.args = args
        self.data_path = data_path
        self.realnormal_path = self.args.realnormal_path
        self.rough_root = args.rough_path
        # print(is_train)
        self.stripes = np.load(self.args.stripes_path)
        self.depth_factor = depth_factor
        
        ##### Transforms ######
        self.train_transforms = [
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]
        if args.do_random_crop:
            self.train_transforms.append(
                A.RandomCrop(args.crop_h, args.crop_w)
            )
        # register count for using CutDepth
        if args.use_cut_depth:
            self.count = 0
        self.to_tensor = transforms.ToTensor()
        
        ##### Filenames ######
        self.filenames = readTXT(filenames_path)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        rgb_file, depth_file = self.filenames[idx].split()[:2]
        rgb_path = os.path.join(self.data_path, rgb_file)
        depth_path = os.path.join(self.data_path, depth_file)
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')/self.depth_factor

        normal = None

        if self.args.realsense:
            if self.args.realkp:

                alb_path = rgb_path.replace('.jpg', '_alb.npy')
                alb = np.load(alb_path)

                _, _, image = script_fsk_depth_k0kp(image, alb, depth, self.stripes, self.is_train, (480, 640), idx, True, normal, None, method=self.args.method)

            normal_file = 'normal_' + os.path.basename(self.filenames[idx].split()[0]).replace('.jpg', '.npy')
            normal = np.load(os.path.join(self.realnormal_path, normal_file))
            normal = cv2.resize(normal, (640, 480), interpolation=cv2.INTER_CUBIC)
            # print(normal.shape)
            # print(normal[0, 0, :])
            # exit(0)
            normal = self.to_tensor(normal)
            
            # print(normal.shape)
            # print(normal[:, 0, 0])
            # exit(0)
            
            stripe_file = self.filenames[idx].split()[2]
            stripe_path = os.path.join(self.data_path, stripe_file)
            striped = cv2.imread(stripe_path)
            striped = cv2.cvtColor(striped, cv2.COLOR_RGB2BGR)

        elif self.args.use_stripe:
            densedepth_file = depth_file.replace('sync_depth', 'dense/sync_depth_dense')
            densedpth_path = os.path.join(self.data_path, densedepth_file)

            densedepth = cv2.imread(densedpth_path, cv2.IMREAD_UNCHANGED).astype('float32')/self.depth_factor

            alb_file = rgb_file.replace('.jpg', '_albedoBS1.png')
            alb_path = os.path.join(self.rough_root, alb_file)
            if not self.is_train:
                alb_path = alb_path.replace('cascade1', 'cascade2')

            normal_path = alb_path.replace('_albedoBS1.png', '_normal1.npy')
            rough_path = alb_path.replace('_albedoBS1.png', '_roughBS1.png')

            alb = cv2.imread(alb_path)
            alb = cv2.cvtColor(alb, cv2.COLOR_BGR2RGB)
            alb = cv2.resize(alb, (640, 480), interpolation=cv2.INTER_CUBIC)

            normal = np.load(normal_path)
            normal = cv2.resize(normal, (640, 480), interpolation=cv2.INTER_CUBIC)

            rough = cv2.imread(rough_path, cv2.IMREAD_UNCHANGED)
            rough = cv2.resize(rough, (640, 480), interpolation=cv2.INTER_CUBIC)

            if self.args.method in [6, 7, 9, 10, 11]:
                _, striped, Onorm = script_fsk_depth_k0kp(image, alb, densedepth, self.stripes, self.is_train, (480, 640), idx, True, normal, rough, method=self.args.method)
                if self.args.Onorm_module:
                    striped = Onorm
            else:
                _, striped, freqs = script_fsk_depth(image, alb, densedepth, self.stripes, self.is_train, (480, 640), idx, True, normal, rough)
        else:
            striped = None
            
        # striped = O_norm when method=6
        image, depth, striped = self.apply_transforms(image, depth, striped)

        if self.args.use_stripe:
            return {"image" : image, "depth" : depth, "stripe": striped, "normal": normal}
        else:
            return {"image" : image, "depth" : depth}
        
    def apply_cut_depth(self, image, depth):
        # image must have shape (H, W, C)
        # depth must have shape (H, W)
        H, W, C = image.shape
        if self.count % 4 == 0:
            alpha = random.random()
            beta = random.random()
            p = 0.75

            l = int(alpha * W)
            w = int(max((W - alpha * W) * beta * p, 1))

            image[:, l:l+w, 0] = depth[:, l:l+w]
            image[:, l:l+w, 1] = depth[:, l:l+w]
            image[:, l:l+w, 2] = depth[:, l:l+w]
        self.count += 1
        return image, depth
    
    def apply_transforms(self, image, depth, stripe=None):
        # image must be a numpy array in range [0, 255] and of shape (H, W, C)
        # depth must be a numpy array of shape (H, W)
        if self.is_train:
            image, depth = self.apply_train_transforms(image, depth, stripe)
        else:
            image, depth = self.apply_test_transforms(image, depth)
        image = self.to_tensor(image)
        depth = self.to_tensor(depth)
        if stripe is not None:
            stripe = self.to_tensor(stripe)
        # image is a torch tensor in range [0, 1] and of shape (C, H, W)
        # depth is a torch tensor and of shape (C, H, W)
        return image, depth, stripe
    
    def apply_test_transforms(self, image, depth):
        return image, depth
    
    def apply_train_transforms(self, image, depth, stripe=None):
        H, W, C = image.shape
        # do padding in case image dimensions 
        # are smaller than crop size
        if self.args.do_random_crop:
            if H < self.args.crop_h:
                image = np.concatenate([image, np.zeros((self.args.crop_h-H, W, 3), dtype=np.uint8)], axis=0)
                depth = np.concatenate([depth, np.zeros((self.args.crop_h-H, W),  dtype=np.uint8)], axis=0)
            if W < self.args.crop_w:
                image = np.concatenate([image, np.zeros((self.args.crop_h, self.args.crop_w-W, 3),  dtype=np.uint8)], axis=0)
                depth = np.concatenate([depth, np.zeros((self.args.crop_h, self.args.crop_w-W),  dtype=np.uint8)], axis=0)

        if self.args.use_augment:
            additional_targets = {"depth" : "mask"}
            aug = A.Compose(
                transforms=self.train_transforms,
                additional_targets=additional_targets
            )
            augmented = aug(image=image, depth=depth)
            image, depth = augmented["image"], augmented["depth"]

        if self.args.use_cut_depth:
            image, depth = self.apply_cut_depth(image, depth)
        return image, depth
    

class SUNRGBDDepthDataset(Dataset):
    def __init__(self, args, is_train, filenames_path, data_path, depth_factor):
        super().__init__()
        self.is_train = is_train
        self.args = args
        self.data_path = data_path
        self.rough_root = args.rough_path
        # print(is_train)
        self.stripes = np.load(self.args.stripes_path)
        self.depth_factor = depth_factor

        self.resize = transforms.Resize((480, 640))

        self.cvshape = (640, 480)
        
        ##### Transforms ######
        self.train_transforms = [
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]
        if args.do_random_crop:
            self.train_transforms.append(
                A.RandomCrop(args.crop_h, args.crop_w)
            )
        # register count for using CutDepth
        if args.use_cut_depth:
            self.count = 0
        self.to_tensor = transforms.ToTensor()
        
        ##### Filenames ######
        self.filenames = readTXT(filenames_path)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        rgb_file, depth_file, dense_file = self.filenames[idx].split()[:3]
        rgb_path = os.path.join(self.data_path, rgb_file)
        depth_path = os.path.join(self.data_path, depth_file)
        dense_path = os.path.join(self.data_path, dense_file)

        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')
        dense_resize = cv2.imread(dense_path, cv2.IMREAD_UNCHANGED).astype('float32')

        # print(image.shape)

        image_resize = cv2.resize(image, self.cvshape, interpolation=cv2.INTER_CUBIC)
        # depth = cv2.resize(depth, self.cvshape, interpolation=cv2.INTER_CUBIC)
        dense_resize = cv2.resize(dense_resize, self.cvshape, interpolation=cv2.INTER_CUBIC)

        depth = np.expand_dims(depth, axis=2)
        depth = np.asarray(depth, dtype=np.float32) / 10000.0

        dense_resize = np.asarray(dense_resize, dtype=np.float32) / 10000.0

        if self.args.use_stripe:
            alb_path = rgb_path.replace('.jpg', '_alb.npy')
            alb = np.load(alb_path)

            alb = cv2.resize(alb, self.cvshape, interpolation=cv2.INTER_CUBIC)

            if self.args.use_rough:
                alb_file = rgb_file.replace('.jpg', '_albedoBS1.png')
                alb_path = os.path.join(self.rough_root, alb_file)

                normal_path = alb_path.replace('_albedoBS1.png', '_normal1.npy')
                rough_path = alb_path.replace('_albedoBS1.png', '_roughBS1.png')

                alb = cv2.imread(alb_path)
                alb = cv2.cvtColor(alb, cv2.COLOR_RGB2BGR)
                normal = np.load(normal_path)
                rough  = cv2.imread(rough_path, cv2.IMREAD_UNCHANGED).astype('float32')

                alb = cv2.resize(alb, self.cvshape, interpolation=cv2.INTER_CUBIC)
                normal = cv2.resize(normal, self.cvshape, interpolation=cv2.INTER_CUBIC)
                rough = cv2.resize(rough, self.cvshape, interpolation=cv2.INTER_CUBIC)

                if self.args.method in [6, 7, 9, 10]:
                    _, striped, Onorm = script_fsk_depth_k0kp(image_resize, alb, dense_resize, self.stripes, self.is_train, (480, 640), idx, True, normal, rough, self.args.method)
                    if self.args.Onorm_module:
                        striped = Onorm
                else:
                    _, striped, freqs = script_fsk_depth(image_resize, alb, dense_resize, self.stripes, self.is_train, (480, 640), idx, True, normal, rough)
            else:
                _, striped, freqs = script_fsk_depth(image_resize, alb, dense_resize, self.stripes, self.is_train, (480, 640), idx)
        else:
            striped = None
            
        image, depth, striped = self.apply_transforms(image, depth, striped)
        image = self.resize(image)
        if self.args.use_stripe:
            striped = self.resize(striped)
        # depth = self.resize(depth)

        if self.args.use_stripe:
            return {"image" : image, "depth" : depth, "stripe": striped}
        else:
            return {"image" : image, "depth" : depth}
        
    def apply_cut_depth(self, image, depth):
        # image must have shape (H, W, C)
        # depth must have shape (H, W)
        H, W, C = image.shape
        if self.count % 4 == 0:
            alpha = random.random()
            beta = random.random()
            p = 0.75

            l = int(alpha * W)
            w = int(max((W - alpha * W) * beta * p, 1))

            image[:, l:l+w, 0] = depth[:, l:l+w]
            image[:, l:l+w, 1] = depth[:, l:l+w]
            image[:, l:l+w, 2] = depth[:, l:l+w]
        self.count += 1
        return image, depth
    
    def apply_transforms(self, image, depth, stripe=None):
        # image must be a numpy array in range [0, 255] and of shape (H, W, C)
        # depth must be a numpy array of shape (H, W)
        if self.is_train:
            image, depth = self.apply_train_transforms(image, depth, stripe)
        else:
            image, depth = self.apply_test_transforms(image, depth)
        image = self.to_tensor(image)
        depth = self.to_tensor(depth)
        if stripe is not None:
            stripe = self.to_tensor(stripe)
        # image is a torch tensor in range [0, 1] and of shape (C, H, W)
        # depth is a torch tensor and of shape (C, H, W)
        return image, depth, stripe
    
    def apply_test_transforms(self, image, depth):
        return image, depth
    
    def apply_train_transforms(self, image, depth, stripe=None):
        H, W, C = image.shape
        # do padding in case image dimensions 
        # are smaller than crop size
        if self.args.do_random_crop:
            if H < self.args.crop_h:
                image = np.concatenate([image, np.zeros((self.args.crop_h-H, W, 3), dtype=np.uint8)], axis=0)
                depth = np.concatenate([depth, np.zeros((self.args.crop_h-H, W),  dtype=np.uint8)], axis=0)
            if W < self.args.crop_w:
                image = np.concatenate([image, np.zeros((self.args.crop_h, self.args.crop_w-W, 3),  dtype=np.uint8)], axis=0)
                depth = np.concatenate([depth, np.zeros((self.args.crop_h, self.args.crop_w-W),  dtype=np.uint8)], axis=0)

        if self.args.use_augment:
            additional_targets = {"depth" : "mask"}
            aug = A.Compose(
                transforms=self.train_transforms,
                additional_targets=additional_targets
            )
            augmented = aug(image=image, depth=depth)
            image, depth = augmented["image"], augmented["depth"]

        if self.args.use_cut_depth:
            image, depth = self.apply_cut_depth(image, depth)
        return image, depth


class IBIMSDepthDataset(Dataset):
    def __init__(self, args, is_train, filenames_path, data_path, depth_factor):
        super().__init__()
        self.is_train = is_train
        self.args = args
        self.data_path = data_path
        self.rough_root = args.rough_path
        # print(is_train)
        self.stripes = np.load(self.args.stripes_path)
        self.depth_factor = depth_factor

        self.cvshape = (640, 480)
        
        ##### Transforms ######
        self.train_transforms = [
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]
        if args.do_random_crop:
            self.train_transforms.append(
                A.RandomCrop(args.crop_h, args.crop_w)
            )
        # register count for using CutDepth
        if args.use_cut_depth:
            self.count = 0
        self.to_tensor = transforms.ToTensor()
        
        ##### Filenames ######
        self.filenames = readTXT(filenames_path)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        alb_file = self.filenames[idx].split()[0].replace('.png', '_alb.npy')
        alb_path = os.path.join(self.data_path, alb_file)

        mat_file = self.filenames[idx].split()[0].replace('raw/rgb', 'mat').replace('png', 'mat')
        mat_path = os.path.join(self.data_path, mat_file)
        data = io.loadmat(mat_path)['data']

        image = data['rgb'][0][0]
        depth = data['depth'][0][0]

        mask_invalid = data['mask_invalid'][0][0]
        mask_transp = data['mask_transp'][0][0]

        image = cv2.resize(image, self.cvshape, interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, self.cvshape, interpolation=cv2.INTER_CUBIC)

        mask_missing = depth.copy() # Mask for further missing depth values in depth map
        mask_missing[mask_missing!=0]=1
        
        mask_valid = mask_transp*mask_invalid*mask_missing # Combine masks
        depth = depth*mask_valid

        ori = depth

        if self.args.use_stripe:
            alb = np.load(alb_path)
            alb = cv2.resize(alb, self.cvshape, interpolation=cv2.INTER_CUBIC)

            if self.args.use_rough:
                alb_file = alb_file.replace('_alb.npy', '_albedoBS1.png')
                alb_path = os.path.join(self.rough_root, alb_file)

                normal_path = alb_path.replace('_albedoBS1.png', '_normal1.npy')
                rough_path = alb_path.replace('_albedoBS1.png', '_roughBS1.png')

                alb = cv2.imread(alb_path)
                alb = cv2.cvtColor(alb, cv2.COLOR_RGB2BGR)
                normal = np.load(normal_path)
                rough  = cv2.imread(rough_path, cv2.IMREAD_UNCHANGED).astype('float32')

                alb = cv2.resize(alb, self.cvshape, interpolation=cv2.INTER_CUBIC)
                normal = cv2.resize(normal, self.cvshape, interpolation=cv2.INTER_CUBIC)
                rough = cv2.resize(rough, self.cvshape, interpolation=cv2.INTER_CUBIC)
                
                if self.args.method in [6, 7, 9, 10]:
                    _, striped, Onorm = script_fsk_depth_k0kp(image, alb, depth, self.stripes, self.is_train, (480, 640), idx, True, normal, rough, self.args.method)
                    if self.args.Onorm_module:
                        striped = Onorm
                else:
                    _, striped, freqs = script_fsk_depth(image, alb, depth, self.stripes, self.is_train, (480, 640), idx, True, normal, rough)
            else:
                _, striped, freqs = script_fsk_depth(image, alb, depth, self.stripes, self.is_train, (480, 640), idx)
        else:
            striped = None

        # if np.array_equal(depth, ori):
        #     print(66)
        #     exit(0)
        depth = np.expand_dims(depth, axis=2)

        image, depth, striped = self.apply_transforms(image, depth, striped)

        if self.args.use_stripe:
            return {"image" : image, "depth" : depth, "stripe": striped}
        else:
            return {"image" : image, "depth" : depth}
        
    def apply_cut_depth(self, image, depth):
        # image must have shape (H, W, C)
        # depth must have shape (H, W)
        H, W, C = image.shape
        if self.count % 4 == 0:
            alpha = random.random()
            beta = random.random()
            p = 0.75

            l = int(alpha * W)
            w = int(max((W - alpha * W) * beta * p, 1))

            image[:, l:l+w, 0] = depth[:, l:l+w]
            image[:, l:l+w, 1] = depth[:, l:l+w]
            image[:, l:l+w, 2] = depth[:, l:l+w]
        self.count += 1
        return image, depth
    
    def apply_transforms(self, image, depth, stripe=None):
        # image must be a numpy array in range [0, 255] and of shape (H, W, C)
        # depth must be a numpy array of shape (H, W)
        if self.is_train:
            image, depth = self.apply_train_transforms(image, depth, stripe)
        else:
            image, depth = self.apply_test_transforms(image, depth)
        image = self.to_tensor(image)
        depth = self.to_tensor(depth)
        if stripe is not None:
            stripe = self.to_tensor(stripe)
        # image is a torch tensor in range [0, 1] and of shape (C, H, W)
        # depth is a torch tensor and of shape (C, H, W)
        return image, depth, stripe
    
    def apply_test_transforms(self, image, depth):
        return image, depth
    
    def apply_train_transforms(self, image, depth, stripe=None):
        H, W, C = image.shape
        # do padding in case image dimensions 
        # are smaller than crop size
        if self.args.do_random_crop:
            if H < self.args.crop_h:
                image = np.concatenate([image, np.zeros((self.args.crop_h-H, W, 3), dtype=np.uint8)], axis=0)
                depth = np.concatenate([depth, np.zeros((self.args.crop_h-H, W),  dtype=np.uint8)], axis=0)
            if W < self.args.crop_w:
                image = np.concatenate([image, np.zeros((self.args.crop_h, self.args.crop_w-W, 3),  dtype=np.uint8)], axis=0)
                depth = np.concatenate([depth, np.zeros((self.args.crop_h, self.args.crop_w-W),  dtype=np.uint8)], axis=0)

        if self.args.use_augment:
            additional_targets = {"depth" : "mask"}
            aug = A.Compose(
                transforms=self.train_transforms,
                additional_targets=additional_targets
            )
            augmented = aug(image=image, depth=depth)
            image, depth = augmented["image"], augmented["depth"]

        if self.args.use_cut_depth:
            image, depth = self.apply_cut_depth(image, depth)
        return image, depth

class DIODEDepthDataset(Dataset):
    def __init__(self, args, is_train, filenames_path, data_path, depth_factor):
        super().__init__()
        self.is_train = is_train
        self.args = args
        self.data_path = data_path
        self.rough_root = args.rough_path
        # print(is_train)
        self.stripes = np.load(self.args.stripes_path)
        self.depth_factor = depth_factor

        self.resize = transforms.Resize(480)

        self.cvshape = (640, 480)
        
        ##### Transforms ######
        self.train_transforms = [
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]
        if args.do_random_crop:
            self.train_transforms.append(
                A.RandomCrop(args.crop_h, args.crop_w)
            )
        # register count for using CutDepth
        if args.use_cut_depth:
            self.count = 0
        self.to_tensor = transforms.ToTensor()
        
        ##### Filenames ######
        self.filenames = readTXT(filenames_path)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        rgb_file, depth_file, mask_file = self.filenames[idx].split()[:3]
        rgb_path = os.path.join(self.data_path, rgb_file)
        depth_path = os.path.join(self.data_path, depth_file)
        mask_path = os.path.join(self.data_path, mask_file)

        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        depth = np.load(depth_path)
        mask  = np.load(mask_path).astype(np.uint8)
        mask = mask[:, :, np.newaxis]

        # print(depth.shape)
        # depth = depth * mask

        image_resize = cv2.resize(image, self.cvshape, interpolation=cv2.INTER_CUBIC)
        depth_resize = cv2.resize(depth, self.cvshape, interpolation=cv2.INTER_CUBIC)

        # exit(0)

        if self.args.use_stripe:
            alb_path = rgb_path.replace('.png', '_alb.npy')
            alb = np.load(alb_path)
            alb = cv2.resize(alb, self.cvshape, interpolation=cv2.INTER_CUBIC)

            if self.args.use_rough:
                alb_file = rgb_file.replace('.png', '_albedoBS1.png')
                alb_path = os.path.join(self.rough_root, alb_file)

                normal_path = alb_path.replace('_albedoBS1.png', '_normal1.npy')
                rough_path = alb_path.replace('_albedoBS1.png', '_roughBS1.png')

                alb = cv2.imread(alb_path)
                alb = cv2.cvtColor(alb, cv2.COLOR_RGB2BGR)
                normal = np.load(normal_path)
                rough  = cv2.imread(rough_path, cv2.IMREAD_UNCHANGED).astype('float32')

                alb = cv2.resize(alb, self.cvshape, interpolation=cv2.INTER_CUBIC)
                normal = cv2.resize(normal, self.cvshape, interpolation=cv2.INTER_CUBIC)
                rough = cv2.resize(rough, self.cvshape, interpolation=cv2.INTER_CUBIC)

                if self.args.method in [6, 7, 9, 10]:
                    _, striped, Onorm = script_fsk_depth_k0kp(image_resize, alb, depth_resize, self.stripes, self.is_train, (480, 640), idx, True, normal, rough, self.args.method)
                    if self.args.Onorm_module:
                        striped = Onorm 
                else:
                    _, striped, freqs = script_fsk_depth(image_resize, alb, depth_resize, self.stripes, self.is_train, (480, 640), idx, True, normal, rough)
            else:
                _, striped, freqs = script_fsk_depth(image_resize, alb, depth_resize, self.stripes, self.is_train, (480, 640), idx)
        else:
            striped = None

        # depth = np.expand_dims(depth, axis=2)

        image, depth, striped = self.apply_transforms(image, depth, striped)

        image = self.resize(image)

        if self.args.use_stripe:
            return {"image" : image, "depth" : depth, "stripe": striped}
        else:
            return {"image" : image, "depth" : depth}
        
    def apply_cut_depth(self, image, depth):
        # image must have shape (H, W, C)
        # depth must have shape (H, W)
        H, W, C = image.shape
        if self.count % 4 == 0:
            alpha = random.random()
            beta = random.random()
            p = 0.75

            l = int(alpha * W)
            w = int(max((W - alpha * W) * beta * p, 1))

            image[:, l:l+w, 0] = depth[:, l:l+w]
            image[:, l:l+w, 1] = depth[:, l:l+w]
            image[:, l:l+w, 2] = depth[:, l:l+w]
        self.count += 1
        return image, depth
    
    def apply_transforms(self, image, depth, stripe=None):
        # image must be a numpy array in range [0, 255] and of shape (H, W, C)
        # depth must be a numpy array of shape (H, W)
        if self.is_train:
            image, depth = self.apply_train_transforms(image, depth, stripe)
        else:
            image, depth = self.apply_test_transforms(image, depth)
        image = self.to_tensor(image)
        depth = self.to_tensor(depth)
        if stripe is not None:
            stripe = self.to_tensor(stripe)
        # image is a torch tensor in range [0, 1] and of shape (C, H, W)
        # depth is a torch tensor and of shape (C, H, W)
        return image, depth, stripe
    
    def apply_test_transforms(self, image, depth):
        return image, depth
    
    def apply_train_transforms(self, image, depth, stripe=None):
        H, W, C = image.shape
        # do padding in case image dimensions 
        # are smaller than crop size
        if self.args.do_random_crop:
            if H < self.args.crop_h:
                image = np.concatenate([image, np.zeros((self.args.crop_h-H, W, 3), dtype=np.uint8)], axis=0)
                depth = np.concatenate([depth, np.zeros((self.args.crop_h-H, W),  dtype=np.uint8)], axis=0)
            if W < self.args.crop_w:
                image = np.concatenate([image, np.zeros((self.args.crop_h, self.args.crop_w-W, 3),  dtype=np.uint8)], axis=0)
                depth = np.concatenate([depth, np.zeros((self.args.crop_h, self.args.crop_w-W),  dtype=np.uint8)], axis=0)

        if self.args.use_augment:
            additional_targets = {"depth" : "mask"}
            aug = A.Compose(
                transforms=self.train_transforms,
                additional_targets=additional_targets
            )
            augmented = aug(image=image, depth=depth)
            image, depth = augmented["image"], augmented["depth"]

        if self.args.use_cut_depth:
            image, depth = self.apply_cut_depth(image, depth)
        return image, depth


class HYPERDepthDataset(Dataset):
    def __init__(self, args, is_train, filenames_path, data_path, depth_factor):
        super().__init__()
        self.is_train = is_train
        self.args = args
        self.data_path = data_path
        self.rough_root = args.rough_path
        # print(is_train)
        # self.stripes = np.load(self.args.stripes_path)
        self.depth_factor = depth_factor
        self.stripes = np.load(self.args.stripes_path)

        self.cvshape = (640, 480)

        self.resize = transforms.Resize((480, 640))
        
        ##### Transforms ######
        self.train_transforms = [
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]
        if args.do_random_crop:
            self.train_transforms.append(
                A.RandomCrop(args.crop_h, args.crop_w)
            )
        # register count for using CutDepth
        if args.use_cut_depth:
            self.count = 0
        self.to_tensor = transforms.ToTensor()
        
        ##### Filenames ######
        self.filenames = readTXT(filenames_path)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        rgb_file, depth_file= self.filenames[idx].split()[:2]
        rgb_path = os.path.join(self.data_path, rgb_file)
        depth_path = os.path.join(self.data_path, depth_file)

        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        with h5py.File(depth_path, 'r') as f:
            depth = np.array(f['dataset'])
            depth = self.depth_transform(depth)

        image = image.astype(np.uint8)
        # depth = depth.astype(np.float32)

        image_resize = cv2.resize(image, self.cvshape, interpolation=cv2.INTER_CUBIC)
        depth_resize = cv2.resize(depth, self.cvshape, interpolation=cv2.INTER_CUBIC)

        if self.args.use_stripe:
            # alb_path = rgb_path.replace('.png', '_alb.npy')
            # alb = np.load(alb_path)
            # alb = cv2.resize(alb, self.cvshape, interpolation=cv2.INTER_CUBIC)

            if self.args.use_rough:
                alb_file = rgb_file.replace('.jpg', '_albedoBS1.png')
                alb_path = os.path.join(self.rough_root, alb_file)

                normal_path = alb_path.replace('_albedoBS1.png', '_normal1.npy')
                rough_path = alb_path.replace('_albedoBS1.png', '_roughBS1.png')

                alb = cv2.imread(alb_path)
                alb = cv2.cvtColor(alb, cv2.COLOR_RGB2BGR)
                normal = np.load(normal_path)
                rough  = cv2.imread(rough_path, cv2.IMREAD_UNCHANGED).astype('float32')

                alb = cv2.resize(alb, self.cvshape, interpolation=cv2.INTER_CUBIC)
                normal = cv2.resize(normal, self.cvshape, interpolation=cv2.INTER_CUBIC)
                rough = cv2.resize(rough, self.cvshape, interpolation=cv2.INTER_CUBIC)

                if self.args.method in [6, 7, 9, 10]:
                    _, striped, Onorm = script_fsk_depth_k0kp(image_resize, alb, depth_resize, self.stripes, self.is_train, (480, 640), idx, True, normal, rough, self.args.method)
                    if self.args.Onorm_module:
                        striped = Onorm 
                else:
                    _, striped, freqs = script_fsk_depth(image_resize, alb, depth_resize, self.stripes, self.is_train, (480, 640), idx, True, normal, rough)
            else:
                _, striped, freqs = script_fsk_depth(image_resize, alb, depth_resize, self.stripes, self.is_train, (480, 640), idx)
        else:
            striped = None

        depth = np.expand_dims(depth, axis=2)

        image, depth, striped = self.apply_transforms(image, depth, striped)

        image = self.resize(image)

        if self.args.use_stripe:
            return {"image" : image, "depth" : depth, "stripe": striped}
        else:
            return {"image" : image, "depth" : depth}
        
    def apply_cut_depth(self, image, depth):
        # image must have shape (H, W, C)
        # depth must have shape (H, W)
        H, W, C = image.shape
        if self.count % 4 == 0:
            alpha = random.random()
            beta = random.random()
            p = 0.75

            l = int(alpha * W)
            w = int(max((W - alpha * W) * beta * p, 1))

            image[:, l:l+w, 0] = depth[:, l:l+w]
            image[:, l:l+w, 1] = depth[:, l:l+w]
            image[:, l:l+w, 2] = depth[:, l:l+w]
        self.count += 1
        return image, depth

    def depth_transform(self, npyDistance):
        intWidth, intHeight, fltFocal = 1024, 768, 886.81

        npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
            1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
        npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                    intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
        npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
        npyImageplane = np.concatenate(
            [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

        npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
        return npyDepth
    
    def apply_transforms(self, image, depth, stripe=None):
        # image must be a numpy array in range [0, 255] and of shape (H, W, C)
        # depth must be a numpy array of shape (H, W)
        if self.is_train:
            image, depth = self.apply_train_transforms(image, depth, stripe)
        else:
            image, depth = self.apply_test_transforms(image, depth)
        image = self.to_tensor(image)
        depth = self.to_tensor(depth)
        if stripe is not None:
            stripe = self.to_tensor(stripe)
        # image is a torch tensor in range [0, 1] and of shape (C, H, W)
        # depth is a torch tensor and of shape (C, H, W)
        return image, depth, stripe
    
    def apply_test_transforms(self, image, depth):
        return image, depth
    
    def apply_train_transforms(self, image, depth, stripe=None):
        H, W, C = image.shape
        # do padding in case image dimensions 
        # are smaller than crop size
        if self.args.do_random_crop:
            if H < self.args.crop_h:
                image = np.concatenate([image, np.zeros((self.args.crop_h-H, W, 3), dtype=np.uint8)], axis=0)
                depth = np.concatenate([depth, np.zeros((self.args.crop_h-H, W),  dtype=np.uint8)], axis=0)
            if W < self.args.crop_w:
                image = np.concatenate([image, np.zeros((self.args.crop_h, self.args.crop_w-W, 3),  dtype=np.uint8)], axis=0)
                depth = np.concatenate([depth, np.zeros((self.args.crop_h, self.args.crop_w-W),  dtype=np.uint8)], axis=0)

        if self.args.use_augment:
            additional_targets = {"depth" : "mask"}
            aug = A.Compose(
                transforms=self.train_transforms,
                additional_targets=additional_targets
            )
            augmented = aug(image=image, depth=depth)
            image, depth = augmented["image"], augmented["depth"]

        if self.args.use_cut_depth:
            image, depth = self.apply_cut_depth(image, depth)
        return image, depth