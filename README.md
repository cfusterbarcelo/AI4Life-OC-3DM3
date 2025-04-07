# AI4Life-OC-3DM3
Python tools for stitching, denoising (Noise2Void), and analyzing 5D microscopy data of tumor spheroids in collagen. Developed as part of the AI4Life Open Call 3 project.

---

## 🧪 Environment Setup (Windows + NVIDIA GPU)

This repo uses [Noise2Void (n2v)](https://github.com/juglab/n2v) for denoising.

To replicate the working GPU-enabled environment on Windows (with TensorFlow 2.10.1), you can use the included `n2v_env.yaml`:

```bash
conda env create -f n2v_env.yaml
conda activate n2v
```

If you're using a **different system** or encounter issues, check the official [n2v installation guide](https://github.com/juglab/n2v#installation) for up-to-date instructions.
