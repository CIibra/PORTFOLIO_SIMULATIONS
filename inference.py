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
