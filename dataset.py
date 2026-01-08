# dataset.py
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import glob
import os

class CT3DSegDataset(Dataset):
    def __init__(self, images_list, masks_list, augment=False):
        self.images = images_list
        self.masks = masks_list
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = nib.load(self.images[idx]).get_fdata().astype(np.float32)
        msk = nib.load(self.masks[idx]).get_fdata().astype(np.uint8)
        # normalize intensity: simple clipping + minmax to [-1,1]
        img = np.clip(img, -1000, 400)
        img = (img - (-1000)) / (1400)  # 0..1
        img = img*2 - 1
        img = np.expand_dims(img, 0)  # channel dim
        msk = np.expand_dims(msk, 0).astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=2).copy()
                msk = np.flip(msk, axis=2).copy()
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=3).copy()
                msk = np.flip(msk, axis=3).copy()
        return torch.from_numpy(img), torch.from_numpy(msk)

def get_file_lists(data_dir="data_synth"):
    imgs = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    masks = sorted(glob.glob(os.path.join(data_dir, "masks", "*_mask.nii.gz")))
    # naive match by filenames prefix
    mapping = []
    for im in imgs:
        base = os.path.basename(im).split(".")[0]
        # find corresponding mask
        mask_candidates = [m for m in masks if os.path.basename(m).startswith(base)]
        if mask_candidates:
            mapping.append((im, mask_candidates[0]))
    # split by filename prefix (train/val/test)
    train = [pair for pair in mapping if "train" in pair[0]]
    val = [pair for pair in mapping if "val" in pair[0]]
    test = [pair for pair in mapping if "test" in pair[0]]
    return train, val, test
#Étape 5 — Modèle 3D U-Net minimal (model.py)