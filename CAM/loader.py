import random
import h5py
import torch
import numpy as np


'''This file based is based on and modified from the tutorial img_sgm'''
class H5ImageLoader:
    def __init__(self, img_file, batch_size, seg_file=None):
        self.img_h5 = h5py.File(img_file, 'r')
        self.dataset_list = list(self.img_h5.keys())

        if seg_file is not None:
            self.seg_h5 = h5py.File(seg_file, 'r')
            if not set(self.dataset_list).issubset(set(self.seg_h5.keys())):
                raise ValueError("Images are not consistent with segmentation.")
        else:
            self.seg_h5 = None

        self.num_images = len(self.img_h5)
        self.batch_size = batch_size
        self.num_batches = self.num_images // self.batch_size  # skip the remainders
        self.img_ids = list(range(self.num_images))
        self.image_size = self.img_h5[self.dataset_list[0]][()].shape

    def __iter__(self):
        self.batch_idx = 0
        random.shuffle(self.img_ids)
        return self

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration

        batch_img_ids = self.img_ids[self.batch_idx * self.batch_size: (self.batch_idx + 1) * self.batch_size]
        datasets = [self.dataset_list[idx] for idx in batch_img_ids]

        # === Load and convert to tensors ===
        images = np.stack([self.img_h5[ds][()] for ds in datasets])
        images = torch.tensor(images).permute(0, 3, 1, 2).float() / 255.0  # (B, C, H, W)

        if self.seg_h5 is not None:
            labels = np.stack([self.seg_h5[ds][()] == 1 for ds in datasets])
            labels = torch.tensor(labels).long()  # (B, H, W)
        else:
            labels = None

        species = torch.tensor([self.img_h5[ds].attrs['species'] for ds in datasets], dtype=torch.long)
        breeds = torch.tensor([self.img_h5[ds].attrs['breed'] for ds in datasets], dtype=torch.long)

        self.batch_idx += 1
        return images, labels, species, breeds
