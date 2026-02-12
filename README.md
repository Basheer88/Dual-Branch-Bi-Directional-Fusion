# Dual-Branch Bi-Directional Fusion for Robust ECG Classification

This repository contains the reference implementation for the paper:

**Dual-Branch Bi-directional Fusion of Temporal and Hierarchical Features for Robust ECG Classification**

The method integrates:
- **Temporal branch:** multi-scale CNN + SE + BiLSTM for rhythm-level temporal dynamics
- **Hierarchical branch:** **CapsNet** + multi-head self-attention for part–whole morphological structure
- **Fusion:** **Adaptive Multi-head Bi-Directional Attention Fusion (AMBAF)** for beat-wise, bidirectional cross-branch interaction
- **Loss:** **Enhanced Adaptive Margin Context Loss (EAMCL)** for class-imbalance mitigation via class-aware adaptive margins

---

## Repository Structure

- `BiDirectionalAttention.py`  
  Implementation of **AMBAF** (bi-directional cross-branch attention / reweighting).

- `BiLSTM.py`  
  Temporal feature extractor (CNN–SE–BiLSTM branch).

- `CapsNet.py`  
  Hierarchical feature extractor (CapsNet branch, including routing and attention if applicable).

- `EAMCL.py`  
  Implementation of **EAMCL** (class-aware adaptive margin loss).

- `config.py`  
  Central configuration (paths, hyperparameters, training settings).

- `LICENSE`  
  License file.

---

## Requirements

> Please update versions as needed to match your environment.

- Python >= 3.8
- NumPy
- SciPy
- scikit-learn
- PyTorch (recommended) **or** TensorFlow/Keras (depending on your implementation)
- matplotlib (optional, for plots)

Example (PyTorch environment):
```bash
pip install numpy scipy scikit-learn matplotlib
# install PyTorch from the official selector: https://pytorch.org/get-started/locally/
```

---

## Datasets

This project uses publicly available ECG benchmark datasets:

- **MIT-BIH Arrhythmia Database** (AAMI evaluation recommended)
- **PTB Diagnostic ECG Database**
- **INCART Database**
- **Sudden Cardiac Death Holter Database (SVDB)**

⚠️ **Important:** Please prepare the datasets **exactly following the dataset preparation protocol described in the paper**
(preprocessing, segmentation/windowing, label mapping, and the inter-patient / cross-dataset split rules).
This is required to reproduce the reported results.

### Dataset Preparation (follow the paper)

Because dataset formats differ, you must convert the raw records into a unified format (e.g., `.npy`, `.pt`, `.mat`, or pre-segmented windows)
using the same steps reported in the manuscript. Typical steps include:

1. Download datasets from the official sources (and comply with usage terms)
2. Apply the same filtering / normalization used in the paper (if any)
3. Segment signals into the same fixed-length windows (or beat-centered segments) used in the paper
4. Map labels to the target taxonomy (e.g., **AAMI 5-class** for MIT-BIH)
5. Save the prepared splits for reproducible training/testing

Configure dataset paths and preprocessing options in `config.py`.

---

## How to Run

> This repository provides the core model components (temporal branch, hierarchical branch, fusion, and loss).
> The recommended training workflow is:
> **(1) prepare data (paper protocol) → (2) train each branch → (3) train/run AMBAF bidirectional fusion**.

### 1) Configure
Edit `config.py` to set:
- dataset paths
- sampling rate / segment length
- classes and label mapping
- training parameters (batch size, epochs, lr)
- evaluation protocol (inter-patient split, cross-dataset transfer)

### 2) Train the Temporal Branch (CNN–SE–BiLSTM)
Train the low-level temporal branch implemented in `BiLSTM.py` using the prepared dataset splits.

Example:
```bash
python train_temporal.py --config config.py
```

Outputs (recommended):
- `checkpoints/temporal_branch.pt`
- training logs

### 3) Train the Hierarchical Branch (CapsNet + Attention)
Train the high-level hierarchical branch implemented in `CapsNet.py`.

Example:
```bash
python train_hierarchical.py --config config.py
```

Outputs (recommended):
- `checkpoints/hierarchical_branch.pt`
- training logs

### 4) Train / Run the Bi-Directional Fusion (AMBAF)
Load the pretrained branch checkpoints and train the fusion module in `BiDirectionalAttention.py`
(end-to-end or fusion-only, according to your paper settings). Use `EAMCL.py` as the training loss if enabled.

Example:
```bash
python train_fusion.py --config config.py   --temporal_ckpt checkpoints/temporal_branch.pt   --hierarchical_ckpt checkpoints/hierarchical_branch.pt
```

### 5) Evaluation and Transfer
Evaluate on the target test split and (optionally) perform cross-dataset transfer (PTB / INCART / SVDB) following the paper protocol.

Example:
```bash
python eval.py --config config.py --checkpoint checkpoints/fusion_model.pt
```

✅ If your training/evaluation scripts use different names, update the commands above to match your filenames.


---

## Reproducibility Notes

To reproduce paper-level results, ensure:
- Fixed random seed (Python/NumPy/framework)
- Same inter-patient split protocol (for MIT-BIH)
- Same window length, sampling rate, and label mapping
- Same training schedule and early-stopping rules (if used)
- Same transfer-learning setup for PTB / INCART / SVDB (linear probing vs fine-tuning)

---

## Configuration (AMBAF + EAMCL)

Key components:
- **AMBAF:** bidirectional, beat-wise cross-branch interaction (see `BiDirectionalAttention.py`)
- **EAMCL:** class-aware adaptive margins (see `EAMCL.py`)

Tuneable parameters (typical):
- number of attention heads
- fusion temperature / scaling (if implemented)
- EAMCL margin base and class-frequency scaling
- optimizer and learning rate schedule

All tunables should be centralized in `config.py`.

---

## License

See `LICENSE`.

---

## Citation

If you use this code in academic work, please cite the paper:

```bibtex
@article{dualbranch_bidirfusion_ecg,
  title   = {Dual-Branch Bi-directional Fusion of Temporal and Hierarchical Features for Robust ECG Classification},
  author  = {Hassoon, Basheer A. and Xiong, Shengwu and Hasson, Mushtaq A. and Salahudeen, Ridwan and Khan, Tauqeer},
  journal = {Measurement},
  year    = {2026},
  note    = {Under review}
}
```

---

## Contact

For questions, please open an Issue in this repository.
