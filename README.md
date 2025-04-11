# AI4Life-OC-3DM3

Python pipeline for stitching, denoising (Noise2Void), and segmenting migrating cells in 5D microscopy data (X, Y, Z, T, C) of tumor spheroids embedded in collagen.  
Developed as part of the AI4Life Open Call 3 project, building on [OC-1 Project 11](https://github.com/ai4life-opencalls/oc-1-project-11/tree/main).

---

## 1. Environment Setup (Windows + NVIDIA GPU)

This project uses:

- [Noise2Void (n2v)](https://github.com/juglab/n2v) for denoising  
- [Cellpose](https://github.com/MouseLand/cellpose) for segmentation

To replicate the GPU-enabled setup:

```
conda env create -f n2v_env.yaml
conda activate n2v
```

This uses:
- `tensorflow==2.10.1` (GPU)
- `cudatoolkit=11.2`
- `cudnn=8.1.0`

For alternate setups, refer to the [n2v installation guide](https://github.com/juglab/n2v#installation).

---

## 2. Stitching Raw Tile Images

We start with tile-based 5D microscopy data (2x2 grid).  
Each tile is a TIFF file with shape (X, Y, Z, T, C).

We stitch them into a single 5D image using:

**File to run:**  
```
stitch.py
```
This script loads each tile using known offsets and saves a single `.ome.tif` with full 5D dimensions preserved.

---

## 3. Preparing 2D+T Data for Denoising

From the stitched 5D data, we extract the nuclei channel and perform a max Z-projection over central slices.  
The result is a 3D (T, Y, X) stack, ready for denoising.

**File to run:**  
```
prepare_n2v.py
```
Output is saved to:

```
data/n2v_input/{N}_slices/*.tif
```
Where `{N}` is the number of projected slices.

---

## 4. Denoising with Noise2Void

We use a Jupyter notebook to train a self-supervised N2V model and denoise each timepoint.

**Notebook to open:**  
```
notebooks/N2V_denoising.ipynb
```
Steps:
- Load projected 3D stack
- Train N2V model using patch-based augmentation
- Predict denoised output
- Save denoised TIFF to disk

Model is saved to:

```
models/denoising_5slices_n2v/
```
We also compute and save image difference plots:

```
models/denoising_5slices_n2v/original-denoised-difference.png
```
---

## 5. Annotating & Training Segmentation with Cellpose

### 5.1 Manual Annotation (done using Cellpose GUI)

We extract individual timepoints from the denoised TIFF stack (done manually or with Fiji).  
We then open selected frames in the **Cellpose GUI**, annotate them, and save masks as `_seg.npy`.

This produces training pairs like:

```
segmentation_input/training/
├── nuclei_t003.tif
├── nuclei_t003_seg.npy
```
Repeat for 5–10 diverse timepoints.

---

### 5.2 Training Custom Cellpose Model

We use a batch file to launch training from these annotation pairs.

**File to run:**  
```
scripts/train_cellpose.bat
```
This runs Cellpose training with:
- base model: `nuclei`
- 50 epochs
- batch size 8
- grayscale channel (chan=0)

Trained model is saved to:

```
models/cellpose_nuclei_model/nuclei_custom/
```
---

## 6. File Structure Overview

```
AI4Life-OC-3DM3/
│
├── models/
│   ├── denoising_5slices_n2v/
│   │   ├── weights_best.h5
│   │   └── original-denoised-difference.png
│   └── cellpose_nuclei_model/
│       └── nuclei_custom/
│
├── data/
│   ├── raw/                           # Raw stitched tiles
│   ├── n2v_input/                     # Projected TIFFs for N2V
│   └── segmentation_input/
│       └── training/                  # .tif + _seg.npy pairs
│
├── notebooks/
│   └── N2V_denoising.ipynb
│
├── scripts/
│   └── train_cellpose.bat
│
├── stitch.py
├── prepare_n2v.py
├── n2v_env.yaml
└── README.md
```
---

Made by the OC-3DM³ team.
