import os
import numpy as np
from scipy.ndimage import gaussian_filter 
from skimage.morphology import ball as sphere
import nibabel as nib
import random

OUT_DIR = "data_synth"
os.makedirs(OUT_DIR, exist_ok=True)

def make_volume(shape=(64,64,64), n_nodules=1, min_rad=3, max_rad=8, seed=None):
    if seed is not None:
        np.random.seed(seed)
    vol = np.random.normal(loc=-900, scale=25, size=shape)  # air-like HU noise
    mask = np.zeros(shape, dtype=np.uint8)

    for _ in range(n_nodules):
        r = np.random.randint(min_rad, max_rad+1)
        z = np.random.randint(r, shape[0]-r)
        y = np.random.randint(r, shape[1]-r)
        x = np.random.randint(r, shape[2]-r)
        # create sphere mask
        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = (zz - z)**2 + (yy - y)**2 + (xx - x)**2
        sph = dist <= r*r
        intensity = np.random.randint(-300, 150)  # soft-tissue to calcified
        vol[sph] = intensity + np.random.normal(0, 5, size=sph.sum())
        mask[sph] = 1

    # smooth a bit to mimic scanner blur
    vol = gaussian_filter(vol, sigma=0.5)
    return vol.astype(np.float32), mask.astype(np.uint8)

def save_nifti(arr, path):
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(img, path)

def generate_dataset(n_samples=200, out_dir=OUT_DIR, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    splits = {"train": int(0.7*n_samples), "val": int(0.15*n_samples), "test": n_samples - int(0.7*n_samples) - int(0.15*n_samples)}
    idx = 0
    split_map = []
    for split, count in splits.items():
        for i in range(count):
            n_nod = np.random.choice([0,1,2,3], p=[0.2,0.5,0.2,0.1])
            vol, mask = make_volume(n_nodules=n_nod, seed=seed+idx)
            img_path = os.path.join(out_dir, "images", f"{split}_{idx:03d}.nii.gz")
            mask_path = os.path.join(out_dir, "masks", f"{split}_{idx:03d}_mask.nii.gz")
            save_nifti(vol, img_path)
            save_nifti(mask, mask_path)
            split_map.append((split, img_path, mask_path))
            idx += 1
    print(f"Generated {idx} volumes -> train/val/test = {splits}")
    return split_map

if __name__ == "__main__":
    generate_dataset(n_samples=200)