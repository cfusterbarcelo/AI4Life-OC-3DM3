"""
prepare_n2v.py

Prepare 3D time-lapse volumes for N2V:
- Extracts channel 0 (nuclei)
- Max-projects middle Z-slices
- Saves [T, Y, X] per scene

Author: @cfusterbarcelo
Creation Date: 04/04/2025

Usage:
    python preprocessing/prepare_n2v.py
"""

import os
from utils import process_for_n2v

INPUT_DIR = "D:/Data/Spheroids-Data-OCProject/Individual_Images/stitched"
OUTPUT_DIR = "D:/Data/Spheroids-Data-OCProject/Individual_Images/n2v_input"
NUM_SLICES = 10

def main():
    slice_str = f"{NUM_SLICES}slices"
    output_dir = os.path.join(OUTPUT_DIR, slice_str)
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".ome.tif"):
            continue

        input_path = os.path.join(INPUT_DIR, fname)
        base_name = os.path.splitext(fname)[0]
        output_name = f"{base_name}_nuclei_TYX_{NUM_SLICES}z.tif"
        output_path = os.path.join(output_dir, output_name)

        process_for_n2v(input_path, output_path, channel_idx=0, num_slices=NUM_SLICES)

if __name__ == "__main__":
    main()
