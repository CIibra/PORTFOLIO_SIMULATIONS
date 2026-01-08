# eval.py
import torch
from dataset import get_file_lists, CT3DSegDataset
from torch.utils.data import DataLoader
from model import UNet3D
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def visualize_case(img_path, mask_path, pred_mask):
    img = nib.load(img_path).get_fdata()
    true_m = nib.load(mask_path).get_fdata()
    # show middle axial slice
    z = img.shape[0] // 2
    fig, axes = plt.subplots(1,3, figsize=(12,4))
    axes[0].imshow(img[z,:,:], cmap='gray'); axes[0].set_title("Image (axial)")
    axes[1].imshow(true_m[z,:,:], cmap='gray'); axes[1].set_title("True mask")
    axes[2].imshow(pred_mask[z,:,:], cmap='gray'); axes[2].set_title("Pred mask")
    plt.show()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_list = get_file_lists("data_synth")
    test_imgs = [p for p,_ in test_list]; test_masks = [m for _,m in test_list]
    ds = CT3DSegDataset(test_imgs, test_masks, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    model = UNet3D(in_ch=1, out_ch=1, base_ch=16).to(device)
    model.load_state_dict(torch.load("checkpoints/unet3d_best.pth", map_location=device))
    model.eval()
    import torch.nn.functional as F
    dices = []
    for i, (x,y) in enumerate(loader):
        x = x.to(device).float()
        y = y.to(device).float()
        with torch.no_grad():
            out = model(x)
            prob = torch.sigmoid(out).cpu().numpy()[0,0]
            pred = (prob > 0.5).astype(np.uint8)
            # compute dice
            pflat = pred.flatten(); yflat = y.numpy().flatten()
            inter = (pflat * yflat).sum()
            dice = (2*inter + 1e-6) / (pflat.sum() + yflat.sum() + 1e-6)
            dices.append(dice)
        if i < 5:
            visualize_case(test_imgs[i], test_masks[i], pred)
    print("Mean Dice on test:", np.mean(dices))

if __name__ == "__main__":
    main()
#Étape 8 — Inference simple (inference.py)
# inference.py
import torch, nibabel as nib, numpy as np
from model import UNet3D
from dataset import CT3DSegDataset, get_file_lists

def infer_one(image_path, model_path="checkpoints/unet3d_best.pth", device="cpu"):
    img = nib.load(image_path).get_fdata().astype(np.float32)
    img_c = np.clip(img, -1000, 400)
    img_c = (img_c - (-1000)) / 1400
    img_c = img_c*2 - 1
    x = torch.from_numpy(np.expand_dims(img_c, (0,1))).float().to(device)  # [1,1,D,H,W]
    model = UNet3D(1,1,16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        out = model(x)
        prob = torch.sigmoid(out).cpu().numpy()[0,0]
        pred = (prob > 0.5).astype(np.uint8)
    # save mask
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine=np.eye(4)), "pred_mask.nii.gz")
    print("Saved pred_mask.nii.gz")

if __name__ == "__main__":
    _,_,test = get_file_lists("data_synth")
    if len(test)>0:
        infer_one(test[0][0], device=("cuda" if torch.cuda.is_available() else "cpu"))
