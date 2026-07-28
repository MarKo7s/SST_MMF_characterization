# Arbitrary Coherence Length Source, Multimode Fibre Characterization by Spatial State Tomography

## Intro

Referenceless, arbitrary coherence source multimode fibre characterization using spatial state tomography. The modules contain functions to perform spatial state tomography and retrieve the complete complex mode transmission matrix (MTM). Those functions are shown in the available Jupyter notebook examples.

## Install

This repository is self-contained. Custom packages live under `MODULES/`. Large experimental arrays (`.npy`) are stored with **Git LFS**.

### 1. Install Git LFS

Install once on your machine: [https://git-lfs.github.com](https://git-lfs.github.com)

Then enable it:

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

Do **not** use GitHub’s “Download ZIP” for the experimental data: ZIP archives contain only LFS pointer stubs, not the real `.npy` files.

### 3. Verify experimental data

After a successful LFS pull, each intensity file should be about **111 MB** (not a few hundred bytes of text). Example check:

```bash
# PowerShell
Get-Item .\experimental_data\030423_5_mode_groups_BW_40nm_1300nm_N_118_wav\BW_0nm\V_0_0.npy | Select-Object Length

# Python
python -c "import numpy as np; a=np.load(r'experimental_data/030423_5_mode_groups_BW_40nm_1300nm_N_118_wav/BW_0nm/V_0_0.npy'); print(a.shape, a.dtype)"
```

If `git lfs pull` fails with an LFS budget / quota error, the hosted binaries are temporarily unavailable via GitHub LFS. Contact the maintainer below for an alternate copy, or use a local backup of the same folder layout.

Expected layout (6 bandwidth folders × `H_0_0.npy` + `V_0_0.npy`, plus `measurement_specs.pkl` per folder):

```
experimental_data/030423_5_mode_groups_BW_40nm_1300nm_N_118_wav/
  BW_0nm/
  BW_5nm/
  BW_10nm/
  BW_20nm/
  BW_30nm/
  BW_40nm/
SST_setup/StokesTomagraphySetUp_projections_1770.pkl
```

### 4. Python dependencies

Required:

- `numpy`, `scipy`, `pathlib` (stdlib), Jupyter / IPython for the notebooks

Optional (acceleration; simulation notebook can fall back to CPU):

- `cupy`, `numba`

Example:

```bash
pip install numpy scipy jupyter matplotlib ipywidgets
# optional:
# pip install numba cupy
```

Add the local modules to the Python path (the notebooks already do this):

```python
import os, sys
sys.path.append(os.path.join(os.getcwd(), "MODULES"))
```

## Notebooks

| Notebook | Needs experimental `.npy` data? | Description |
|----------|----------------------------------|-------------|
| `SST_MTM_retrival_example_simulation.ipynb` | No | End-to-end MTM retrieval on simulated data |
| `SST_MTM_retrival_example_experimental.ipynb` | Yes | Retrieval on measured Stokes intensity sweeps |

## Contact

For more data availability and bug reports: m.maestremorote@uq.edu.au
