Disclaimer: For rewriting parts of this README.md ChatGPT has been used. The exact prompt was: How to generate a READme.md file for this (with the file attached).


# **SATELLITE INPAINTING & SUPER-RESOLUTION** #

This project studies image inpainting and single-image super-resolution for satellite RGB imagery.
All experiments, models, and results are contained in a single Jupyter notebook.

**Repository**:

GitLab:
https://gitlab.ewi.tudelft.nl/dsait4120/student-repositories/2025-2026/finalproject_16

**Overview**:

The notebook implements two consecutive pipelines:

Inpainting of missing or corrupted regions using a residual U-Net with a PatchGAN discriminator

Super-resolution (SRGAN-style) applied to the inpainted images to recover fine spatial detail

The super-resolution stage is intentionally applied after inpainting, reflecting a realistic satellite-processing workflow.

**Code structure** (algorithmic pointers):

All code is located in:

Super_resolution_and_inpainting_Stan.ipynb


**Key components:**

Connected hole mask generation (controlled coverage)
build_connected_hole_mask() and connected_blob_mask()
→ used to generate realistic, spatially coherent missing regions

Inpainting losses (masked)
L1 reconstruction loss, gradient loss, and ring loss
Implemented in residual_inpaint_loss(...)

Inpainting inference
Holes are filled using zero-valued corruption to match the real data distribution

Diffusion baseline
Stable Diffusion inpainting (runwayml/stable-diffusion-inpainting)
Used as a qualitative reference method

Super-resolution dataset creation
Synthetic low-resolution images created via bicubic downsampling and upsampling
Implemented in SRSatelliteCached(Dataset)

Super-resolution model
Residual U-Net generator with PatchGAN discriminator
Trained using a MixGE loss (MSE + gradient loss) with GAN warmup

**Environment setup**

The notebook was developed and tested in Google Colab.
Required packages can be installed with:

pip install torch torchvision torchaudio
pip install diffusers transformers accelerate safetensors
pip install numpy pillow matplotlib tqdm tensorboard

**Running the project**

Open and run the notebook from top to bottom:

jupyter notebook Super_resolution_and_inpainting_Stan.ipynb


No command-line arguments are required.
All paths and parameters are defined inside the notebook.

Notes

Normalization layers are intentionally avoided in the SR generator, following common practice in super-resolution literature (e.g. EDSR).

GAN losses are introduced only after a reconstruction warmup phase for stability.

Visual evaluation focuses on zoomed-in regions, where improvements over bicubic upsampling are most visible.