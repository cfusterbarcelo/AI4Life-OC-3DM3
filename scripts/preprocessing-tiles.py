import numpy as np
import os
from skimage import io, filters
from skimage.morphology import white_tophat, disk

# ====== CONFIGURATION ======
DATA_DIR = "D:/Data/Spheroids-Data-OCProject/Individual_Images/background-subs"
# DATA_DIR = "Path/to/my/images"
SAVE_DIR = "D:/Data/Spheroids-Data-OCProject/Individual_Images/stitched-after-BS-10S"

# Specify the axis order of the images after stitching (must match what your tiles have)
AXIS_ORDER = "TCZYX"  # Options: "TCZYX", "TZYX", etc.

# ====== Z-PROJECTION SETTINGS ======
DO_Z_PROJECTION = True

Z_NUM_SLICES = 5  # Number of slices to project (same for all)

# Define starting Z-slice per scene (use the stitched scene name)
Z_START_PER_SCENE = {
    "l_2_2_20X-03-Scene-01": 10,
    "l_2_2_20X-03-Scene-02": 17,
    "l_2_2_20X-03-Scene-03": 11,
    # Add more scenes here as needed
}

TILE_POSITIONS = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

def get_nuclei_channel(arr, axis_order):
    c_axis = axis_order.index('C')
    slicer = [slice(None)] * arr.ndim
    slicer[c_axis] = 0
    return arr[tuple(slicer)]

def estimate_flatfield(tile_stack):
    # tile_stack: ZYX or ZYXC numpy array
    flatfield_slice = tile_stack[0]  # Take the first z-slice (assumed least cells)
    flatfield = filters.gaussian(flatfield_slice, sigma=50, preserve_range=True)
    flatfield = np.clip(flatfield, 1, None)  # Avoid division by zero
    return flatfield

def apply_flatfield_correction(tile_stack, flatfield):
    corrected = tile_stack / flatfield  # Broadcasting over Z, Y, X
    return corrected

def subtract_background(tile_stack, radius=50):
    # Apply background subtraction per z-slice
    result = np.zeros_like(tile_stack)
    for z in range(tile_stack.shape[0]):
        result[z] = white_tophat(tile_stack[z], disk(radius))
    return result

def z_max_projection_per_time(tile_stack, z_start, z_num_slices):
    T, Z, Y, X = tile_stack.shape
    projected_stack = np.zeros((T, Y, X), dtype=tile_stack.dtype)
    for t in range(T):
        z_end = z_start + z_num_slices
        z_end = min(z_end, Z)
        projected_stack[t] = np.max(tile_stack[t, z_start:z_end, :, :], axis=0)
    return projected_stack

def stitch_2x2_tiles_per_time(tile_projs):
    T, Y, X = tile_projs[0].shape
    stitched_stack = np.zeros((T, 2*Y, 2*X), dtype=tile_projs[0].dtype)
    for t in range(T):
        stitched_stack[t, :Y, :X]     = tile_projs[0][t]  # Top-left
        stitched_stack[t, :Y, X:]     = tile_projs[1][t]  # Top-right
        stitched_stack[t, Y:, :X]     = tile_projs[2][t]  # Bottom-left
        stitched_stack[t, Y:, X:]     = tile_projs[3][t]  # Bottom-right
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
        for pos in TILE_POSITIONS:
            # Example filename pattern -- update as needed!
            filename = f"{scene}_{pos}.tif"
            path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Tile file not found: {path}")
            tile_files.append(path)
        
        for tile_path in tile_files:
            # --- LOAD IMAGE ---
            img = io.imread(tile_path)
            # img shape: e.g. (T, C, Z, Y, X) or (T, Z, C, Y, X)
            # Adjust this reshape/squeeze block to match your true shape and axis order
            img = np.squeeze(img)  # Remove single dimensions if needed

            # --- AXIS REORDER ---
            # Make sure axes are in TCZYX order for easy handling
            axes = list(AXIS_ORDER)
            tgt_order = [axes.index(ax) for ax in 'TCZYX' if ax in axes]
            img = np.transpose(img, tgt_order)

            # --- SELECT NUCLEI CHANNEL (assume C=0) ---
            img = get_nuclei_channel(img, 'TCZYX')  # now img shape (T, Z, Y, X)

            # --- FLATFIELD CORRECTION (per timepoint, per tile) ---
            # Optional: Do flatfield per T? If not, just use Z0 from T0.
            flatfield = estimate_flatfield(img[0])  # Just use first T as estimate
            img_ff = np.zeros_like(img)
            for t in range(img.shape[0]):
                img_ff[t] = apply_flatfield_correction(img[t], flatfield)

            # --- BACKGROUND SUBTRACTION (per Z slice) ---
            img_bs = np.zeros_like(img_ff)
            for t in range(img_ff.shape[0]):
                img_bs[t] = subtract_background(img_ff[t], radius=50)

            # --- Z-PROJECTION (per timepoint) ---
            tile_proj = z_max_projection_per_time(img_bs, z_start, Z_NUM_SLICES)  # shape (T, Y, X)
            tile_projs.append(tile_proj)

        # --- STITCH THE FOUR TILES ---
        stitched_stack = stitch_2x2_tiles_per_time(tile_projs)  # (T, Y, X)

        # --- SAVE OUTPUT ---
        out_path = os.path.join(SAVE_DIR, f"{scene}_stitched.tif")
        save_stack_as_tiff(stitched_stack, out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()