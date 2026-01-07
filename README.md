# No-Clean-Reference Image Super-Resolution: Application to Electron Microscopy 

This repository provides the official implementation of **No-Clean-Reference Electron Microscopy Super-Resolution (EMSR)** — a deep learning framework for reconstructing high-resolution (HR) 3D electron microscopy images from noisy low-resolution (LR) acquisitions over large brain tissue volumes.

EMSR enables **practical, scalable, and high-quality EM super-resolution**, even in real-world settings where obtaining clean high-resolution reference images is difficult or impossible.  
Our approach is specifically tailored for large-field-of-view brain imaging, providing a robust solution for high-throughput EM reconstruction without requiring clean ground-truth data.


---

## Dataset Structure (Training)

Training data should follow the structure below:
```

DATASET_ROOT/
└── Train/
├── HR/ # High-resolution reference (or pseudo-GT)
│ ├── SampleA/
│ │ ├── SampleA_000001.png
│ │ ├── SampleA_000101.png
│ │ └── ...
│ └── SampleB/
│ ├── SampleB_000001.png
│ └── ...
└── LR/ # Low-resolution inputs
├── SampleA/
│ ├── SampleA_000001.png
│ ├── SampleA_000101.png
│ └── ...
└── SampleB/
├── SampleB_000001.png
└── ...
'''
