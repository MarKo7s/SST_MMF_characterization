# Broadband Control of Light through Complex Media via Autonomously Self-Referencing Transmission Matrix Characterisation

**Version:** [0.1.0](https://github.com/MarKo7s/SST_MMF_characterization/releases/tag/v0.1.0) (arXiv snapshot) · **Dataset DOI:** [10.5281/zenodo.21637820](https://doi.org/10.5281/zenodo.21637820)

## Intro

Self-referenced, arbitrary coherence source multimode fibre characterization using spatial state tomography. The SST framework lives under `MODULES/`; the Jupyter notebooks at the repository root are end-to-end **examples** that call those modules to perform spatial state tomography and retrieve the complete complex mode transmission matrix (MTM).

## Notebooks Examples:


| Notebook                                      | Needs experimental `.npy` data? | Description                                   |
| --------------------------------------------- | ------------------------------- | --------------------------------------------- |
| `SST_MTM_retrival_example_simulation.ipynb`   | No                              | End-to-end MTM retrieval on simulated data    |
| `SST_MTM_retrival_example_experimental.ipynb` | Yes                             | Retrieval on measured Stokes intensity sweeps |




## Repository structure

```
SST_MMF_characterization/
├── MODULES/                          # SST framework (importable library code)
│   ├── stokes/                       # Tomography, MTM retrieval, analysis
│   │   ├── stokestomography.py
│   │   ├── retrivalMTM.py
│   │   ├── processingTools.py
│   │   ├── analyzefucTools.py
│   │   └── PSD.py
│   ├── fibremodes/                   # Mode basis and TM helpers
│   │   ├── ModesGen.py
│   │   ├── mode_generation_core_library.py
│   │   └── transmission_matrix_generator.py
│   └── custom_plotting.py
├── SST_MTM_retrival_example_*.ipynb  # Example notebooks (use MODULES/)
├── experimental_data/                # Measured intensities (Git LFS or Zenodo)
├── SST_setup/                        # Analyser projection metadata
├── environment.yml / requirements.txt
└── README.md
```

- `MODULES/` — the reusable SST / MTM characterisation framework.
- **Root notebooks** — worked examples; they add `MODULES/` to `sys.path` and demonstrate the full pipeline.
- `experimental_data/` — large `.npy` arrays (Git LFS; Zenodo fallback below).
- `SST_setup/` — Stokes analyser projection pickle used by the experimental example.



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

Core packages: `numpy`, `scipy`, `matplotlib`, `numba`, `numexpr`, `einsumt`, Jupyter / `ipywidgets`, `ipyparallel`.

**GPU (CuPy) is required for practical runtimes** of the example notebooks. Install the wheel that matches your CUDA toolkit (defaults in `environment.yml` / `requirements.txt` use CUDA 13.x for this lab setup):

```bash
pip install cupy-cuda13x
# or: pip install cupy-cuda12x
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

**If Git LFS fails** (quota, missing LFS client, or Download ZIP): download the experimental dataset from Zenodo ([DOI: 10.5281/zenodo.21637820](https://doi.org/10.5281/zenodo.21637820)), unpack `experimental_data.zip` at the repository root so paths match:

```
experimental_data/030423_5_mode_groups_BW_40nm_1300nm_N_118_wav/
  BW_0nm/ ... BW_40nm/   # H_0_0.npy, V_0_0.npy, measurement_specs.pkl
SST_setup/StokesTomagraphySetUp_projections_1770.pkl
```



### 5. CPU vs GPU notebook flags

Defaults assume a CUDA GPU (`GPU=True`, `engine='GPU'`) and are the recommended path for performance. On CPU-only machines (much slower):

- Simulation notebook: `engine='CPU', multicore=False` for `LGmodes`, and `StokesVectorCalc(..., GPU=False)`
- Experimental notebook: `Stokes_Tomography_MTM_retrival(..., GPU=False)` (LG engine follows this flag)

`multicore=True` with `engine='CPU'` requires `ipyparallel` and a running `ipcluster`.

## Versioning

This repository is tagged for manuscript snapshots. **`v0.1.0`** is the arXiv-associated release; later tags (e.g. `v0.2.0`) may follow journal revisions. Prefer a release tag over an untagged `main` tip when citing the code.

## Contact

For more data availability and bug reports: [m.maestremorote@uq.edu.au](mailto:m.maestremorote@uq.edu.au)