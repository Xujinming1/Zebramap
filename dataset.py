import random
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os
from synthesis import script_fsk_depth_k0kp
class DepthDataset(Dataset):
    def __init__(self, args, is_train, filenames_path, data_path, depth_factor):
        super().__init__()
        self.is_train = is_train
        self.args = args
        self.data_path = data_path
        self.realnormal_path = self.args.realnormal_path
        self.normal_path = args.normal_path
        self.stripes = np.load(self.args.stripes_path)
        self.depth_factor = depth_factor

        self.shape = (640, 480)

        self.to_tensor = transforms.ToTensor()
        
        ##### Filenames ###### 
        with open(filenames_path, 'r') as f:
            self.filenames = f.readlines()
        
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

        if not self.args.Onorm_module:
            normal_file = 'normal_' + os.path.basename(self.filenames[idx].split()[0]).replace('.jpg', '.npy')
            normal = np.load(os.path.join(self.realnormal_path, normal_file))
            normal = cv2.resize(normal, self.shape, interpolation=cv2.INTER_CUBIC)
            normal = self.to_tensor(normal)
            
            stripe_file = self.filenames[idx].split()[2]
            stripe_path = os.path.join(self.data_path, stripe_file)
            striped = cv2.imread(stripe_path)
            striped = cv2.cvtColor(striped, cv2.COLOR_RGB2BGR)

        else:
            densedepth_file = depth_file.replace('sync_depth', 'dense/sync_depth_dense')
            densedpth_path = os.path.join(self.data_path, densedepth_file)

            densedepth = cv2.imread(densedpth_path, cv2.IMREAD_UNCHANGED).astype('float32')/self.depth_factor

            alb_file = rgb_file.replace('.jpg', '_albedoBS1.png')
            alb_path = os.path.join(self.normal_path, alb_file)
            if not self.is_train:
                alb_path = alb_path.replace('cascade1', 'cascade2')

            normal_path = alb_path.replace('_albedoBS1.png', '_normal1.npy')

            alb = cv2.imread(alb_path)
            alb = cv2.cvtColor(alb, cv2.COLOR_BGR2RGB)
            alb = cv2.resize(alb, self.shape, interpolation=cv2.INTER_CUBIC)

            normal = np.load(normal_path)
            normal = cv2.resize(normal, self.shape, interpolation=cv2.INTER_CUBIC)

            _, striped, Onorm = script_fsk_depth_k0kp(image, alb, densedepth, self.stripes, self.is_train, (480, 640), idx, normal, self.args.Onorm_module)
            if self.args.Onorm_module:
                striped = Onorm
            
        # striped = O_norm when method=6
        image = self.to_tensor(image)
        depth = self.to_tensor(depth)
        striped = self.to_tensor(striped)

        return {"image" : image, "depth" : depth, "stripe": striped, "normal": normal}
        