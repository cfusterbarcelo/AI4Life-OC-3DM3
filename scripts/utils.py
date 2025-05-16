"""
utils.py

Utility functions for stitching and preprocessing of 5D microscopy images
(C, Z, T, Y, X) for the AI4Life-OC-3DM3 project.

Author: @cfusterbarcelo
Creation Date: 04/04/2025

Sections:
- Stitching utilities
- N2V preparation utilities (for later use)
"""

import os
import numpy as np
import tifffile
import re
from collections import defaultdict

# -----------------------------
# 📌 Stitching Utilities
# -----------------------------

TILE_OFFSETS = {
    0: (0, 0),
    1: (684, 0),
    2: (0, 648),
    3: (684, 648),
}

def reorder_to_tczyx(img):
    """
    Reorder axes to [T, C, Z, Y, X].

    Supports:
    - 5D input [Z, T, C, Y, X]
    - 4D input [Z, T, Y, X] (adds dummy channel)
    """
    if img.ndim == 5:  # [Z, T, C, Y, X]
        return img.transpose(1, 2, 0, 3, 4)
    elif img.ndim == 4:  # [Z, T, Y, X] → add C=1
        img = img.transpose(1, 0, 2, 3)         # → [T, Z, Y, X]
        img = img[:, np.newaxis, :, :, :]      # → [T, C=1, Z, Y, X]
        return img
    else:
        raise ValueError(f"Unexpected input shape: {img.shape}")

def load_tile(path):
    """Load a single 5D image tile and reorder axes."""
    raw = tifffile.imread(path)  # shape: [Z, T, C, Y, X]
    return reorder_to_tczyx(raw)

def compute_stitched_shape(tiles, offsets):
    """Compute the shape of the stitched image from all tiles."""
    t, c, z, y, x = next(iter(tiles.values())).shape
    max_x = max(offset[0] + x for offset in offsets)
    max_y = max(offset[1] + y for offset in offsets)
    return (t, c, z, max_y, max_x)

def stitch_tiles(tile_paths):
    """Stitch 4 tiles into one large 5D image."""
    tiles = {
        TILE_OFFSETS[i]: load_tile(tile_paths[i])
        for i in range(4)
    }

    stitched_shape = compute_stitched_shape(tiles, TILE_OFFSETS.values())
    stitched = np.zeros(stitched_shape, dtype=np.uint16)

    for (ox, oy), tile in tiles.items():
        _, _, _, h, w = tile.shape
        stitched[:, :, :, oy:oy+h, ox:ox+w] = tile

    return stitched

def find_scenes(data_dir):
    """Group tile paths by scene prefix."""
    scene_groups = defaultdict(dict)
    pattern = re.compile(r"(l_2_2_20X-03-Scene-\d{2})_M-(\d)\.tif")

    for fname in os.listdir(data_dir):
        match = pattern.match(fname)
        if match:
            scene, tile_idx = match.groups()
            scene_groups[scene][int(tile_idx)] = fname

    return scene_groups

# -----------------------------
# 📌 N2V Preparation Utilities
# -----------------------------

def extract_channel(volume, channel_idx=0):
    """Extract a single channel from a 5D volume [T, C, Z, Y, X]."""
    return volume[:, channel_idx, :, :, :]  # [T, Z, Y, X]

def max_project_middle_slices(volume, num_slices=10):
    """Max project the middle Z-slices of a 4D stack [T, Z, Y, X]."""
    t, z, y, x = volume.shape
    z_start = (z - num_slices) // 2
    z_end = z_start + num_slices
    return volume[:, z_start:z_end, :, :].max(axis=1)  # [T, Y, X]

def process_for_n2v(input_path, output_path, channel_idx=0, num_slices=10):
    """
    Process a stitched 5D volume for N2V:
    - Load stitched [T, C, Z, Y, X]
    - Extract specified channel
    - Max project central Z-slices
    - Save as 3D [T, Y, X]
    """
    import tifffile
    import numpy as np

    print(f"🔍 Processing {os.path.basename(input_path)}")
    volume = tifffile.imread(input_path)              # [T, C, Z, Y, X]
    channel = extract_channel(volume, channel_idx)    # [T, Z, Y, X]
    projected = max_project_middle_slices(channel, num_slices)  # [T, Y, X]
    tifffile.imwrite(output_path, projected.astype(np.uint16))
    print(f"✅ Saved: {output_path}")
