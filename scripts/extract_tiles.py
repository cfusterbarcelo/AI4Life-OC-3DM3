"""
extract_tiles.py

each tile of a stitched czi file is extracted and
saved as a separate tif

Author: FME
Creation Date: 15/04/2025
updated: 23/05/2025

"""

import aicspylibczi
import numpy as np
import tifffile
import os

INPUT_DIR = r'C:\Users\ac132194\Documents\Fiona_Data\test' #path of the czi file
OUTPUT_DIR = r'C:\Users\ac132194\Documents\Fiona_Data\test\processed'# path where the stacks should be saved

SPACING = 4.0 # spacing in Z direction
XY_RESOLUTION = 0.454 #x and y resoultion


def main():
    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".czi"):
            continue
        
        input_path = os.path.join(INPUT_DIR, fname)
        base_name = os.path.splitext(fname)[0]

        czi = aicspylibczi.CziFile(input_path)
        dimensions = czi.get_dims_shape()

        for tile in range(0, dimensions[0]["M"][1]):
            empty_img = np.zeros((dimensions[0]["T"][1], dimensions[0]["Z"][1], dimensions[0]["C"][1], dimensions[0]["Y"][1], dimensions[0]["X"][1]), dtype='uint16')
            print(f"🔄 working on tile {tile}...")
            for time in range(0, dimensions[0]["T"][1]):
                for z_slice in range(0, dimensions[0]["Z"][1]):
                    for channel in range(0, dimensions[0]["C"][1]):
                        czi_img, shp = czi.read_image(C=channel, S=0, Z=z_slice, T=time, M=tile) # read only the specified image
                        czi_np_xy = czi_img[0, 0, 0, 0, 0, 0, :, :] # transfortm to 2D np array
                        empty_img[time, z_slice, channel] += czi_np_xy # the empty_img array is then filled with the array from the image

            save_path = os.path.join(OUTPUT_DIR, f"{base_name}_M_{tile}.tif")

            tifffile.imwrite(save_path,  
                    empty_img, 
                    imagej=True,
                    resolution=(1.0 / XY_RESOLUTION, 1.0 / XY_RESOLUTION), 
                    metadata={
                        'spacing': SPACING, 
                        'unit': 'um',
                        'axes': 'TZCYX' #imagej stack need this order 
                        }
                    )
            print(f"✅ Saved tile {tile} to {save_path}")


if __name__ == "__main__":
    main()