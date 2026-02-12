# Complex Applied Image Processing Pipeline for Satellite Imagery 
## Setup

Use python dependency manager, like uv (https://github.com/astral-sh/uv) or conda. For uv:

```bash
uv venv .venv --python 3.11 # Create virtual environment
uv sync # Install all dependencies (from pyproject.toml)
```
To add new dependencies:

```bash
uv add torch
```
## Externals

### Height Estimation & Multi-day 3D Point Cloud K-means Reconstruction

Due to the files being in NTIF and working with an RPC camera model and using propiertary DigitalGlobe metadata, we have to use external tools to pre-process the data.
An attempt was made to use python libraries instead, but nothing worked as well as the external tools. The external tools are only used **to crop and align the data**, bypassing the need deal with the RPC camera model itself.
- gdal -> Image cropping and conversion to tiffs, for the different processing steps. https://gdal.org/en/stable/download.html.
- Ames Stereo Pipeline -> for image alignment. https://stereopipeline.readthedocs.io/en/latest/installation.html
  Extract to `external/` in the project root (e.g., `external/StereoPipeline-3.6.0-x86_64-Linux/`)

If a _gdal.so not error is thrown when running the viewer, it means that gdal libraries required to use the Python API are not compiled (can happend with the pre-compiled binaries). The following commands should fix it (replace brew with your system package):
```bash
export LDFLAGS="-L$(brew --prefix gdal)/lib"
export CPPFLAGS="-I$(brew --prefix gdal)/include"
uv cache clean gdal
uv sync
```



## Data

Sample images for each member are included (via Git LFS), but not the full dataset. Model checkpoints must be downloaded for some components to work.

**https://huggingface.co/datasets/MJ22x/satellite-image-restoration-data**

```bash
huggingface-cli download MJ22x/satellite-image-restoration-data --repo-type dataset --local-dir data/
```

The data/ directory should look exactly like at the link above

### Full 3D Point Cloud Dataset

The full dataset is available on aws, we can use the aws cli to download it.
```bash
aws s3 ls s3://spacenet-dataset/Hosted-Datasets/mvs_dataset
aws s3 sync s3://spacenet-dataset/Hosted-Datasets/mvs_dataset/WV3/ data/WV3/
```


## Run the viewer

```bash
uv run viewer.py
```

To run a component, choose one from the menu on the right and it will add new layers.
BE CAREFUL to not load too many layers at once or it will crash.


## Individual components

As per project requirements, everyone's own components and contributions are in members/member_name. 

### 3D Point Cloud
If you would wish to overlap the point cloud togheter with the heigh map in the 3d viewer of napari, this is possible by selecting the "Height Map" layer on the left panel and the running 

If the scale is set automatically the 2d view becomes skewed sadly, this was the cleanest solution found after some research.
```python
viewer.layers.selection.active
layer.scale = (2.0, 1.0)
```
