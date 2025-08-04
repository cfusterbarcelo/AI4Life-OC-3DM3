import os
import numpy as np
from tifffile import imread, imsave
from skimage.transform import resize

# === PARAMETER ANPASSEN ===
input_dir = r'C:\Users\ac132194\Documents\Fiona_Data\Tiles\3D\CAREamics-output\test\train'             # Ordner mit .tif + _seg.npy
output_dir = r'C:\Users\ac132194\Documents\Fiona_Data\Tiles\3D\CAREamics-output\test\train\train_slices'      # Zielordner für 2D-Slices
anisotropy = 8.81                 # Z-zu-XY Verhältnis (z. B. 8.81)
xz_step = 5                       # z. B. 1 = alle, 5 = jeder 5. Y-Slice
yz_step = 5                       # z. B. 1 = alle, 5 = jeder 5. X-Slice

os.makedirs(output_dir, exist_ok=True)

def resize_zaxis(img, axis=0, factor=1.0):
    if factor == 1.0:
        return img
    shape = list(img.shape)
    shape[axis] = int(round(shape[axis] * factor))
    return resize(img, shape, order=0, preserve_range=True, anti_aliasing=False).astype(img.dtype)

def has_mask_pixels(mask_slice):
    return np.any(mask_slice > 0)

for fname in os.listdir(input_dir):
    if fname.endswith('.tif'):
        name = fname[:-4]
        img = imread(os.path.join(input_dir, fname))               # shape: [Z, Y, X]
        seg_path = os.path.join(input_dir, f"{name}_seg.npy")
        seg_dict = np.load(seg_path, allow_pickle=True).item()
        seg = seg_dict['masks']                                    # shape: [Z, Y, X]

        # === XY-Slices (immer alle) ===
        for z in range(img.shape[0]):
            if not has_mask_pixels(seg[z]):
                continue
            imsave(f"{output_dir}/{name}_z{z:03d}_XY.tif", img[z])
            imsave(f"{output_dir}/{name}_z{z:03d}_XY_masks.tif", seg[z])

        # === XZ-Slices (entlang Y) ===
        for y in range(0, img.shape[1], xz_step):
            mask_xz = seg[:, y, :]
            if not has_mask_pixels(mask_xz):
                continue
            img_xz = resize_zaxis(img[:, y, :], axis=0, factor=anisotropy)
            mask_xz = resize_zaxis(mask_xz, axis=0, factor=anisotropy)
            imsave(f"{output_dir}/{name}_y{y:03d}_XZ.tif", img_xz)
            imsave(f"{output_dir}/{name}_y{y:03d}_XZ_masks.tif", mask_xz)

        # === YZ-Slices (entlang X) ===
        for x in range(0, img.shape[2], yz_step):
            mask_yz = seg[:, :, x]
            if not has_mask_pixels(mask_yz):
                continue
            img_yz = resize_zaxis(img[:, :, x], axis=0, factor=anisotropy)
            mask_yz = resize_zaxis(mask_yz, axis=0, factor=anisotropy)
            imsave(f"{output_dir}/{name}_x{x:03d}_YZ.tif", img_yz)
            imsave(f"{output_dir}/{name}_x{x:03d}_YZ_masks.tif", mask_yz)

print("✅ Slicing abgeschlossen.")