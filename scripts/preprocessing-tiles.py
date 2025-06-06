import numpy as np
import os
from skimage import io, filters
from skimage.morphology import white_tophat, disk
import time
from skimage.filters import gaussian
from skimage import exposure

start = time.time()
# ====== CONFIGURATION ======
DATA_DIR = "D:/Data/Spheroids-Data-OCProject/Individual_Images/originals"
# DATA_DIR = "Path/to/my/images"
SAVE_DIR = "D:/Data/Spheroids-Data-OCProject/Individual_Images/preprocessed-by-tiles"

# Specify the axis order of the images after stitching (must match what your tiles have)
AXIS_ORDER = "TZCYX"  # Options: "TCZYX", "TZYX", etc.

# ====== Z-PROJECTION SETTINGS ======
DO_Z_PROJECTION = True

Z_NUM_SLICES = 7  # Number of slices to project (same for all)

# Define starting Z-slice per scene (use the stitched scene name)
Z_START_PER_SCENE = {
    "l_2_2_20X-03-Scene-01": 10,
    "l_2_2_20X-03-Scene-02": 17,
    "l_2_2_20X-03-Scene-03": 11,
    # Add more scenes here as needed
}
TILE_SUFFIXES = ['M-0', 'M-1', 'M-2', 'M-3']

def get_nuclei_channel(arr, axis_order):
    c_axis = axis_order.index('C')
    slicer = [slice(None)] * arr.ndim
    slicer[c_axis] = 0
    return arr[tuple(slicer)]

def estimate_flatfield(tile_stack):
    # tile_stack: ZYX or ZYXC numpy array
    flatfield_slice = tile_stack[-1]  # Take the last z-slice (assumed least cells)
    flatfield = filters.gaussian(flatfield_slice, sigma=50, preserve_range=True)
    flatfield = np.clip(flatfield, 1, None)  # Avoid division by zero
    return flatfield

def apply_flatfield_correction(tile_stack, flatfield):
    corrected = tile_stack / flatfield  # Broadcasting over Z, Y, X
    return corrected

def subtract_background(tile_stack, radius=30):
    # Apply background subtraction per z-slice
    result = np.zeros_like(tile_stack)
    for z in range(tile_stack.shape[0]):
        result[z] = white_tophat(tile_stack[z], disk(radius))
    return result

def fast_background_subtract_stack(tile_stack, sigma=30):
    # tile_stack: (Z, Y, X)
    background = gaussian(tile_stack, sigma=(0, sigma, sigma), preserve_range=True)
    result = tile_stack - background
    return np.clip(result, 0, None)

def z_max_projection_per_time(tile_stack, z_start, z_num_slices):
    T, Z, Y, X = tile_stack.shape
    print(f"  Shape for z-max-proj: (T,Z,Y,X)={tile_stack.shape}, z_start={z_start}, z_num_slices={z_num_slices}")
    projected_stack = np.zeros((T, Y, X), dtype=tile_stack.dtype)
    for t in range(T):
        z_end = min(z_start + z_num_slices, Z)
        if z_end <= z_start:
            print(f"    WARNING: z_end ({z_end}) <= z_start ({z_start}); skipping timepoint {t}")
            continue
        zstack = tile_stack[t, z_start:z_end, :, :]
        if zstack.shape[0] == 0:
            print(f"      ERROR: No Z-slices for timepoint {t}, scene!")
        projected_stack[t] = np.max(zstack, axis=0)
    return projected_stack

def stitch_2x2_tiles_per_time_max(tile_projs):
    T = tile_projs[0].shape[0]
    tile_height, tile_width = tile_projs[0].shape[1], tile_projs[0].shape[2]
    positions = [(0, 0), (684, 0), (0, 648), (684, 648)]

    stitched_height = 648 + tile_height
    stitched_width = 684 + tile_width

    stitched_stack = np.zeros((T, stitched_height, stitched_width), dtype=tile_projs[0].dtype)

    for idx, (x0, y0) in enumerate(positions):
        for t in range(T):
            tile_img = tile_projs[idx][t]
            y1 = y0 + tile_height
            x1 = x0 + tile_width
            # Use max, not sum/average
            stitched_stack[t, y0:y1, x0:x1] = np.maximum(stitched_stack[t, y0:y1, x0:x1], tile_img)
    return stitched_stack

def save_stack_as_tiff(stack, out_path):
    io.imsave(out_path, stack.astype(np.float32), check_contrast=False)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # For each scene
    for scene, z_start in Z_START_PER_SCENE.items():
        print(f'Processing scene {scene}')
        tile_projs = []
        
        # ====== LOAD TILES ======
        # Update the code below to find the correct filenames for each scene and tile position
        tile_files = []
        for pos in TILE_SUFFIXES:
            # Example filename pattern -- update as needed!
            filename = f"{scene}_{pos}.tif"
            path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Tile file not found: {path}")
            tile_files.append(path)
        
        for tile_path in tile_files:
            # --- LOAD IMAGE ---
            img = io.imread(tile_path)
            print(f"  Raw loaded shape: {img.shape}")
            # img shape: e.g. (T, C, Z, Y, X) or (T, Z, C, Y, X)
            # Adjust this reshape/squeeze block to match your true shape and axis order
            img = np.squeeze(img)  # Remove single dimensions if needed
            print(f"  After squeeze: {img.shape}")
            print(f"Step took {time.time() - start:.2f} s")

            # --- AXIS REORDER ---
            # Make sure axes are in TCZYX order for easy handling
            # axes = list(AXIS_ORDER)
            # tgt_order = [axes.index(ax) for ax in 'TCZYX' if ax in axes]
            # img = np.transpose(img, tgt_order)

            # --- SELECT NUCLEI CHANNEL (assume C=0) ---
            img = get_nuclei_channel(img, AXIS_ORDER)  # now img shape (T, Z, Y, X)

            # --- FLATFIELD CORRECTION (per tile, using Z-slice from last T as flatfield) ---
            # flatfield = estimate_flatfield(img[-1])  # img[-1] is the last T, shape (Z, Y, X)
            # img_ff = np.zeros_like(img)  # shape (T, Z, Y, X)
            # for t in range(img.shape[0]):
            #     img_ff[t] = apply_flatfield_correction(img[t], flatfield)
            # print(f"Flatfield correction took {time.time() - start:.2f} s")

            img_ff = img
            # --- FAST BACKGROUND SUBTRACTION (per timepoint, batch over Z) ---
            img_bs = np.zeros_like(img_ff)  # shape (T, Z, Y, X)
            for t in range(img_ff.shape[0]):
                img_bs[t] = fast_background_subtract_stack(img_ff[t], sigma=10)
            print(f"Background subtraction took {time.time() - start:.2f} s")
        
            # # --- Z-PROJECTION (per timepoint) ---
            tile_proj = z_max_projection_per_time(img_bs, z_start, Z_NUM_SLICES)  # shape (T, Y, X)

            tile_projs.append(tile_proj)
            print(f"Z-projection took {time.time() - start:.2f} s")
        # --- STITCH THE FOUR TILES ---
        stitched_stack = stitch_2x2_tiles_per_time_max(tile_projs)  # (T, Y, X)
        print(f"Stitching took {time.time() - start:.2f} s")
        stitched_stack = exposure.rescale_intensity(stitched_stack, in_range='image', out_range=(0, 1))

        # --- SAVE OUTPUT ---
        out_path = os.path.join(SAVE_DIR, f"{scene}_stitched.tif")
        save_stack_as_tiff(stitched_stack, out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()