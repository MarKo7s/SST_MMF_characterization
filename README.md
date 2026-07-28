# Broadband Control of Light through Complex Media via Autonomously Self-Referencing Transmission Matrix Characterisation

## Intro

Self-referenced, arbitrary coherence source multimode fibre characterization using spatial state tomography. The modules contain functions to perform spatial state tomography and retrieve the complete complex mode transmission matrix (MTM). Those functions are shown in the available Jupyter notebook examples.

## Install

This repository is self-contained. Custom packages live under `MODULES/`. Large experimental arrays (`.npy`) are stored with **Git LFS**. If LFS is unavailable, use the Zenodo dataset path below.

### 1. Install Git LFS

Install once on your machine: [https://git-lfs.github.com](https://git-lfs.github.com)

```bash
git lfs install
```

### 2. Clone the repository

```bash
git clone https://github.com/MarKo7s/SST_MMF_characterization.git
cd SST_MMF_characterization
```

If you already cloned without LFS, fetch the large files from the repo root:

```bash
git lfs install
git lfs pull
```

Do **not** use GitHub’s “Download ZIP” for the experimental `.npy` files: that archive contains only LFS pointer stubs.

### 3. Environment (Conda recommended)

From the repository root:

```bash
conda env create -f environment.yml
conda activate sst-mmf
```

Or with pip:

```bash
pip install -r requirements.txt
```

Core packages: `numpy`, `scipy`, `matplotlib`, `numba`, `numexpr`, `einsumt`, Jupyter / `ipywidgets`.

Optional GPU (pick one wheel matching your CUDA toolkit):

```bash
pip install cupy-cuda12x
# or: pip install cupy-cuda11x
```

Register a Jupyter kernel (optional):

```bash
python -m ipykernel install --user --name sst-mmf --display-name "Python (sst-mmf)"
```

The notebooks add `MODULES/` to `sys.path` automatically.

### 4. Experimental data (LFS or Zenodo)

**Preferred:** after a successful clone / `git lfs pull`, each intensity file should be about **111 MB**.

```bash
# PowerShell
Get-Item .\experimental_data\030423_5_mode_groups_BW_40nm_1300nm_N_118_wav\BW_0nm\V_0_0.npy | Select-Object Length

# Python
python -c "import numpy as np; a=np.load(r'experimental_data/030423_5_mode_groups_BW_40nm_1300nm_N_118_wav/BW_0nm/V_0_0.npy'); print(a.shape, a.dtype)"
```

**If Git LFS fails** (quota, missing LFS client, or Download ZIP): download the experimental dataset from **Zenodo** (DOI: *TBD — deposit coming soon*), unpack it at the repository root so paths match:

```
experimental_data/030423_5_mode_groups_BW_40nm_1300nm_N_118_wav/
  BW_0nm/ ... BW_40nm/   # H_0_0.npy, V_0_0.npy, measurement_specs.pkl
SST_setup/StokesTomagraphySetUp_projections_1770.pkl
```

Until the Zenodo record is published, contact the maintainer below for an alternate copy.

### 5. CPU vs GPU notebook flags

Defaults assume a CUDA GPU (`GPU=True`, `engine='GPU'`). On CPU-only machines:

- Simulation notebook: `engine='CPU', multicore=False` for `LGmodes`, and `StokesVectorCalc(..., GPU=False)`
- Experimental notebook: `Stokes_Tomography_MTM_retrival(..., GPU=False)` (LG engine follows this flag)

`multicore=True` with `engine='CPU'` requires `ipyparallel` and a running `ipcluster`.

## Notebooks

| Notebook | Needs experimental `.npy` data? | Description |
|----------|----------------------------------|-------------|
| `SST_MTM_retrival_example_simulation.ipynb` | No | End-to-end MTM retrieval on simulated data |
| `SST_MTM_retrival_example_experimental.ipynb` | Yes | Retrieval on measured Stokes intensity sweeps |

## Contact

For more data availability and bug reports: m.maestremorote@uq.edu.au
