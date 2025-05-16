"""
stitch.py

Stitch 5D image tiles into stitched volumes using utils.py.
Each scene is saved as a 5D OME-TIFF: [T, C, Z, Y, X].

Author: @cfusterbarcelo
Creation Date: 04/04/2025

Usage:
    python preprocessing/stitch.py
"""

import os
import tifffile
from utils import find_scenes, stitch_tiles
import numpy as np

# ====== CONFIGURATION ======
DATA_DIR = "D:/Data/Spheroids-Data-OCProject/Individual_Images/background-subs"
SAVE_DIR = "D:/Data/Spheroids-Data-OCProject/Individual_Images/stitched-after-BS-10S"

# Specify the axis order of the images after stitching (must match what your tiles have)
AXIS_ORDER = "TCZYX"  # Options: "TCZYX", "TZYX", etc.

# ====== Z-PROJECTION SETTINGS ======
DO_Z_PROJECTION = True

Z_NUM_SLICES = 10  # Number of slices to project (same for all)

# Define starting Z-slice per scene (use the stitched scene name)
Z_START_PER_SCENE = {
    "l_2_2_20X-03-Scene-01": 10,
    "l_2_2_20X-03-Scene-02": 17,
    "l_2_2_20X-03-Scene-03": 11,
    # Add more scenes here as needed
}

# =========================================

def z_projection(volume, axis_order, start_slice, num_slices):
    """Performs max Z-projection over a specific range of Z-slices."""
    z_idx = axis_order.index("Z")
    z_size = volume.shape[z_idx]
    end_slice = min(start_slice + num_slices, z_size)

    if start_slice >= z_size or end_slice <= start_slice:
        raise ValueError(f"❌ Invalid Z slice range: {start_slice}:{end_slice} (Z size: {z_size})")

    print(f"Projecting Z slices {start_slice} to {end_slice - 1} (Z size: {z_size})")
    return np.max(volume.take(indices=range(start_slice, end_slice), axis=z_idx), axis=z_idx)



def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    scenes = find_scenes(DATA_DIR)

    for scene_name, tile_files in scenes.items():
        print(f"🔄 Stitching {scene_name}...")

        if len(tile_files) != 4:
            print(f"⚠️ Skipping {scene_name}: expected 4 tiles, found {len(tile_files)}")
            continue

        tile_paths = {
            idx: os.path.join(DATA_DIR, fname)
            for idx, fname in tile_files.items()
        }

        stitched = stitch_tiles(tile_paths)
        print(f"Loaded tile shape: {stitched.shape}")
        print(f"Axis order used: {AXIS_ORDER}")

        # Apply Z-projection if configured
        if "Z" in AXIS_ORDER and DO_Z_PROJECTION:
            if scene_name in Z_START_PER_SCENE:
                z_start = Z_START_PER_SCENE[scene_name]
                print(f"🟢 Applying Z-projection for {scene_name}: start at {z_start}, {Z_NUM_SLICES} slices")
                stitched = z_projection(stitched, AXIS_ORDER, z_start, Z_NUM_SLICES)
                AXES_METADATA = AXIS_ORDER.replace("Z", "")
            else:
                print(f"⚠️ Skipping Z-projection for {scene_name}: no Z_START defined")
                AXES_METADATA = AXIS_ORDER
        else:
            AXES_METADATA = AXIS_ORDER

        save_path = os.path.join(SAVE_DIR, f"{scene_name}.ome.tif")
        tifffile.imwrite(
            save_path,
            stitched,
            photometric='minisblack',
            metadata={'axes': AXES_METADATA},
            ome=True
        )
        print(f"✅ Saved to {save_path}")

if __name__ == "__main__":
    main()
