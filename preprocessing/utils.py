"""
utils.py

Utility functions for stitching and preprocessing of 5D microscopy images
(C, Z, T, Y, X) for the AI4Life-OC-3DM3 project.

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
    """Convert [Z, T, C, Y, X] to [T, C, Z, Y, X] for OME-TIFF export."""
    return img.transpose(1, 2, 0, 3, 4)

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

