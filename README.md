# No-Clean-Reference Image Super-Resolution  

### Application to Electron Microscopy (EMSR)

This repository contains the official implementation of **No-Clean-Reference Electron Microscopy Super-Resolution (EMSR)** — a framework designed for super-resolving electron microscopy (EM) images without requiring perfectly aligned or clean HR targets.  
EMSR enables practical, scalable EM super-resolution for real-world datasets where clean reference images are difficult or impossible to obtain.

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
