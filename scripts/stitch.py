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
from scripts import utils

DATA_DIR = "D:/Data/Spheroids-Data-OCProject/Individual_Images/Process"
SAVE_DIR = "D:/Data/Spheroids-Data-OCProject/Individual_Images/stitched"

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    scenes = utils.find_scenes(DATA_DIR)

    for scene_name, tile_files in scenes.items():
        print(f"🔄 Stitching {scene_name}...")

        if len(tile_files) != 4:
            print(f"⚠️ Skipping {scene_name}: expected 4 tiles, found {len(tile_files)}")
            continue

        tile_paths = {
            idx: os.path.join(DATA_DIR, fname)
            for idx, fname in tile_files.items()
        }

        stitched = utils.stitch_tiles(tile_paths)

        save_path = os.path.join(SAVE_DIR, f"{scene_name}.ome.tif")
        tifffile.imwrite(
            save_path,
            stitched,
            photometric='minisblack',
            metadata={'axes': 'TCZYX'},
            ome=True
        )
        print(f"✅ Saved to {save_path}")

if __name__ == "__main__":
    main()
