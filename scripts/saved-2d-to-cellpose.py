import numpy as np
import tifffile
from pathlib import Path
from shutil import rmtree
import pandas as pd
# import ace_tools as tools

# Define input/output paths
input_dir = Path("D:/Data/Spheroids-Data-OCProject/Individual_Images/zprojection-denoised/")
output_dir = Path("D:/Data/Spheroids-Data-OCProject/Individual_Images/zprojection-denoised-cellpose-training/")
output_dir.mkdir(exist_ok=True)

# Clean old outputs if they exist
if output_dir.exists():
    rmtree(output_dir)
output_dir.mkdir()

# Find all .tif volumes
tif_files = sorted(input_dir.glob("*.tif"))

# Process each volume and extract 2D labeled slices
saved_files = []

for tif_file in tif_files:
    base_name = tif_file.stem
    seg_file = input_dir / f"{base_name}_seg.npy"
    
    if not seg_file.exists():
        continue
    
    volume = tifffile.imread(tif_file)
    mask_dict = np.load(seg_file, allow_pickle=True).item()
    mask = mask_dict["masks"]
    
    for z in range(mask.shape[0]):
        if np.any(mask[z]):
            img_slice = volume[z]
            mask_slice = mask[z]

            slice_name = f"{base_name}_z{z:02d}"
            img_out = output_dir / f"{slice_name}.tif"
            mask_out = output_dir / f"{slice_name}_seg.npy"

            # Compose dictionary in Cellpose format
            dummy_flows = [np.zeros_like(mask_slice, dtype=np.float32) for _ in range(3)]
            dummy_outlines = np.zeros_like(mask_slice, dtype=np.uint8)

            cellpose_mask = {
                "masks": mask_slice.astype(np.uint16),
                "flows": dummy_flows,
                "outlines": dummy_outlines,
            }

            # Save image and corresponding full mask dictionary
            tifffile.imwrite(img_out, img_slice.astype(np.uint16))
            np.save(mask_out, cellpose_mask)

            saved_files.append({"Slice Name": img_out.name, "Type": "Image"})
            saved_files.append({"Slice Name": mask_out.name, "Type": "Mask"})