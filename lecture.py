import nibabel as nib

# Charger le fichier
img = nib.load("pred_mask.nii.gz")

# Récupérer les données sous forme de tableau NumPy
data = img.get_fdata()

# Afficher quelques infos
print("Dimensions :", data.shape)
print("Type :", data.dtype)

# Exemple : accéder à une coupe
import matplotlib.pyplot as plt
plt.imshow(data[:, :, data.shape[2]//2], cmap="gray")
plt.show()
