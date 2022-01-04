"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    def __init__(self, is_train, data_aug_conf, grid_conf):
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.samples = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()


        print(self)

    def prepro(self):
        samples = []
        samples += self.add_scenarios('data', 'ClearNoon_', 34, 314)

        return samples

    def add_scenarios(self, path, scene, frame_begin, frame_end):
        return [{'path': path, 'scene': scene, 'frame': i}
                    for i in range(frame_begin, frame_end + 1)]

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, index, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        img_path = os.path.join(self.samples[index]['path'], self.samples[index]['scene'])
        tf_path = os.path.join(self.samples[index]['path'], 'transformation')
        for cam in cams:
            # read image
            imgname = os.path.join(img_path, cam, "{:08d}".format(self.samples[index]['frame']) + '.png')

            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # read transformation
            tf_name = os.path.join(tf_path, cam + '.txt')
            with open(tf_name, 'r') as fp:
                # intrinsic
                line = fp.readline().rstrip()

                m00, m01, m02, m10, m11, m12, m20, m21, m22 = line.split(" ")
                intrin = torch.Tensor([[float(m00), float(m01), float(m02)], 
                                       [float(m10), float(m11), float(m12)], 
                                       [float(m20), float(m21), float(m22)]])
                # trans
                line = fp.readline().rstrip()
                x, y, z = line.split(" ")
                tran = torch.Tensor([float(x), float(y), float(z)])
                # rot
                line = fp.readline().rstrip()
                w, x, y, z = line.split(" ")
                rot = torch.Tensor(Quaternion([float(w), float(x), float(y), float(z)]).rotation_matrix)

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img.convert('RGB')))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_binimg(self, index):
        bin_path = os.path.join(self.samples[index]['path'], 'segmentation')
        bins = []
        segs = ['cross_walk', 'other_cars', 'white_broken_lane', 
                'yelow_solid_lane', 'drivable_lae', 'shoulder', 
                'white_solid_lane', 'non-drivable_area', 'side_walk', 'yellow_broken_lane']
        for seg in segs:
            bin_name = os.path.join(bin_path, seg, str(self.samples[index]['frame']) + '.npy')
            bin = np.load(bin_name)
            bins.append(bin)

        return torch.Tensor(bins)

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.samples)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):

        return 0

class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):

        cams = ['left', 'front', 'right']
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(index, cams)
        binimg = self.get_binimg(index)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]
    traindata = parser(is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf)
    valdata = parser(is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
