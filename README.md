
# No-Clean-Reference Electron Microscopy Super-Resolution: Application to Electron Microscopy

This repository contains the official implementation of **Electron Microscopy Image Super-Resolution (EMSR)** — a deep-learning framework for reconstructing high-resolution (HR) 3D electron microscopy from noisy low-resolution (LR) acquisitions, without requiring clean ground truth.

EMSR is designed for **large-field-of-view EM imaging** and enables high-quality, scalable SR in realistic EM acquisition scenarios.

---

## Features
- ✔ Supports training with real and synthetic LR/HR pairs  
- ✔ Transformer-based SR designed for EM textures  
- ✔ Full 3D stack inference with sliding-window reconstruction  
- ✔ Automatic configuration + checkpoint management  

---

## Dataset Structure (Training)

Your training data should follow:

```
DATA_ROOT/
└── Train/
    ├── HR/
    │   ├── SampleA/
    │   │   ├── SampleA_000001.png
    │   │   ├── SampleA_000101.png
    │   │   └── ...
    │   └── SampleB/
    │       ├── SampleB_000001.png
    │       └── ...
    │
    └── LR/
        ├── SampleA/
        │   ├── SampleA_000001.png
        │   ├── SampleA_000101.png
        │   └── ...
        └── SampleB/
            ├── SampleB_000001.png
            └── ...
```

---

## 🚀 Installation

### 1. Clone the repository
```
git EMSR.git
cd EMSR
```

### 2. Create and activate the environment
```
conda env create -f environment.yaml
conda activate EMSR
```

### 3. Add EMSR to PYTHONPATH
```
export PYTHONPATH=$(pwd):$PYTHONPATH
```

---

## Training EMSR

All training settings are defined in:

```
configs/train_config.py
```

Edit the following:

- `data.train_dir`  
- `experiment.epochs`  
- `model.*` and `params.*`  

### Start training:
```
python train.py
```

### During training:

- saves the full configuration → `logs/CONFIG.yaml`  
- saves checkpoints → `logs/checkpoints/*.pt`  
- logs TensorBoard → `logs/`  

---

## Inference on 3D EM `.mat` Files

The model loads:

- training config (`logs/CONFIG.yaml`)  
- selected checkpoint (`--ckpt`)  
- inference settings from `configs/inference_config.py`  

### Run inference:
```
python inference.py     --logs_dir /path/to/logs     --mat_dir /path/to/mat_files     --output_dir /path/to/output     --ckpt /path/to/checkpoints/ckpts_step_450.pt
```

### Arguments

- **--logs_dir** → folder containing `CONFIG.yaml`  
- **--mat_dir** → folder of `.mat` stacks  
- **--output_dir** → where SR results (.h5) will be saved  
- **--ckpt** → choose checkpoint manually  

---

## Environment

```
conda env create -f environment.yaml
```

---

## 📚 Citation
If you find this work useful in your research, please consider citing our paper and starring our repository.
```bibtex
@article{khateri2024no,
  title={No-clean-reference image super-resolution: Application to electron microscopy},
  author={Khateri, Mohammad and Ghahremani, Morteza and Sierra, Alejandra and Tohka, Jussi},
  journal={IEEE Transactions on Computational Imaging},
  year={2024},
  publisher={IEEE}
}
```

---

## Acknowledgments
We thank the CSC–IT Center for Science (Finland) and the Bioinformatics Center at the University of Eastern Finland for providing computational resources.

---
