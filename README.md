# Efficient Deep Learning Approach for Diagnosis of ADHD in Children Based on EEG Signals

Classifying children with **Attention-Deficit/Hyperactivity Disorder (ADHD)** versus a healthy control group from **EEG signals**, by turning each EEG epoch into a Morlet time-frequency RGB image and training convolutional and residual neural networks. This repository accompanies the paper published in *Cognitive Computation* (2024).

> ADHD is a common behavioral disorder in children that can persist into adulthood if untreated, so early and reliable diagnosis matters. This work explores whether an image-based deep learning pipeline applied to EEG time-frequency representations can support that diagnosis.

---

## Overview

The pipeline takes raw EEG recordings from two groups of children (an ADHD group and a control group), slices them into short overlapping epochs, and converts each epoch into a colour (RGB) time-frequency image using a **Morlet wavelet** transform. These images are then used to train two classifiers:

1. A **convolutional neural network (CNN)** with dropout regularisation.
2. A **customised residual network (ResNet)** built from scratch with identity and convolutional residual blocks.

Both models are evaluated with **10-fold stratified cross-validation**, at two levels of granularity:

- **Epoch (segment) level** — each EEG segment is classified independently.
- **Subject level** — predictions are aggregated per child to produce a per-subject diagnosis.

---

## Repository structure

```
.
├── ADHD.ipynb            # Original end-to-end Colab notebook (data loading through evaluation)
├── adhd_eeg/              # Reusable Python package, extracted from the notebook
│   ├── config.py          #   dataset path resolution (no more hardcoded Google Drive paths)
│   ├── resnets_utils.py   #   one-hot encoding helper (previously an unincluded external import)
│   ├── features.py        #   EEG -> Morlet-wavelet RGB image feature extraction
│   ├── models.py          #   CNN and customised-ResNet architectures
│   ├── cv.py               #   segment-based train/test splitting and k-fold cross-validation
│   └── train.py            #   command-line entry point
├── tests/
│   └── test_smoke.py       # Import/config/model smoke tests that run without the dataset or a GPU
├── data/
│   └── README.md           # Where to place the dataset (contents are gitignored)
├── requirements.txt         # Runtime dependencies (pinned ranges)
├── requirements-dev.txt      # + pytest, for running the test suite
├── pyproject.toml            # Makes `adhd_eeg` installable (`pip install -e .`)
├── LICENSE
└── README.md                 # This file
```

The notebook is kept as-is for reference and for anyone reproducing results in Colab exactly as published. The `adhd_eeg/` package extracts the same logic into plain, importable, testable Python so the pipeline can also run as a script on any machine (local workstation, server, CI).

---

## Setup

Requires **Python 3.9+**.

```bash
git clone https://github.com/sheperd007/Efficient-Deep-Learning-Approach-for-Diagnosis-of-ADHD-in-Children-Based-on-EEG-Signals.git
cd Efficient-Deep-Learning-Approach-for-Diagnosis-of-ADHD-in-Children-Based-on-EEG-Signals

python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt   # or requirements.txt if you don't need the test suite
pip install -e .                       # makes `adhd_eeg` importable from anywhere
```

### Verify the install (no dataset required)

```bash
pytest tests/test_smoke.py -v
```

This checks that every module imports cleanly, that the CNN and ResNet models build and run a forward pass on synthetic data with the paper's input shape, and that missing-dataset errors are raised with a clear, actionable message. It does **not** require the EEG dataset, a GPU, or network access, so it's a good first thing to run on a fresh machine.

---

## Data

The dataset itself is **not included** in this repository (subject data / licensing). The pipeline expects EEGLAB-format (`.set`/`.fdt`) recordings from two groups of children — ADHD and healthy control — laid out as:

```
<data_dir>/
├── ADHD_part1/*.set
├── ADHD_part2/*.set
├── Control_part1/*.set
└── Control_part2/*.set
```

Obtain the dataset used in the accompanying paper (or a compatible EEGLAB dataset with the same structure) and place it under `data/adhd_dataset/` in this repo, or anywhere else on disk. Point the code at it with either:

- an environment variable: `export ADHD_DATA_DIR=/path/to/adhd_dataset`, or
- a command-line flag: `--data-dir /path/to/adhd_dataset`

If the directory is missing or incomplete, `adhd_eeg.train` and `adhd_eeg.config.resolve_data_dir` fail immediately with a message explaining exactly what's expected — they do not fail silently or produce fabricated results.

For the original Colab workflow, the notebook instead expects the dataset on Google Drive under `MyDrive/adhd_dataset/` with the same subfolder layout.

---

## How to run

### Option A — Python script / CLI (local machine)

```bash
python -m adhd_eeg.train --data-dir /path/to/adhd_dataset --model resnet
```

Key flags (see `python -m adhd_eeg.train --help` for the full list):

| Flag | Default | Meaning |
|------|---------|---------|
| `--data-dir` | `$ADHD_DATA_DIR` or `./data/adhd_dataset` | Dataset location |
| `--model` | `resnet` | `cnn` or `resnet` |
| `--folds` | `10` | Number of cross-validation folds |
| `--epochs` | `100` | Max training epochs per fold (early stopping on `val_accuracy`) |
| `--batch-size` | `128` | Training batch size |
| `--segment-size` | `20` | Epochs per subject-level segment (matches the paper) |

This runs the full Morlet-image feature extraction and k-fold cross-validation described in the paper. It requires the real dataset and can take a long time on CPU — a GPU is recommended for anything beyond a couple of folds.

### Option B — Original Colab notebook

1. Open `ADHD.ipynb` in [Google Colab](https://colab.research.google.com/) (GPU runtime recommended).
2. Place the EEG dataset on your Google Drive under `MyDrive/adhd_dataset/` (EEGLAB `.set` files for the `ADHD_part*` and `Control_part*` groups).
3. Run the cells top to bottom. The notebook mounts Drive, installs dependencies, generates the Morlet RGB images, and trains/evaluates the models.

> Note: `ADHD.ipynb` imports `resnets_utils` (helper functions such as `convert_to_one_hot`), which was a companion file used with the original ResNet code and was never included in this repository. A drop-in reimplementation is provided at `adhd_eeg/resnets_utils.py` — add this repo's root (or just that file) to your Colab path if you hit `ModuleNotFoundError: No module named 'resnets_utils'`.

---

## Methods / Approach

**Signal preprocessing & feature representation**
- EEG recordings are read with [MNE-Python](https://mne.tools/) (`mne.io.read_raw_eeglab`).
- Each recording is segmented into **2-second epochs with 1.5 s overlap** (`mne.make_fixed_length_epochs`).
- Each epoch is transformed into a time-frequency representation with a **Morlet wavelet** (`mne.time_frequency.tfr_morlet`, `n_cycles=5`) and rendered as an **RGB image**, which becomes the model input.

**Models**
- **CNN with dropout** — a stack of `Conv2D` / `MaxPooling2D` layers with dropout for regularisation.
- **Customised ResNet** — a residual network built from scratch using custom `identity_block` and `convolutional_block` functions (the notebook also imports `ResNet50V2` from Keras for reference/comparison).
- Both models use the **Adam** optimiser and **binary cross-entropy** loss.

**Evaluation**
- **10-fold stratified cross-validation** (`StratifiedKFold(n_splits=10, shuffle=True)`) at both the epoch level and the subject level.
- Reported metrics include accuracy, precision, recall, and F1-score, plus confusion matrices (`sklearn.metrics`).

---

## Reproducibility notes

- `adhd_eeg/cv.py` extracts the cross-validation logic in `ADHD.ipynb` line-for-line, including how leftover ("partial") segments are always folded into the training set rather than held out.
- One quirk is preserved deliberately rather than "fixed": the notebook's segment-level `StratifiedKFold` split is stratified against a constant dummy column, not the real class label. Changing this would change the reported cross-validation results, so it's kept as published and called out in a code comment in `adhd_eeg/cv.py`.
- Random seeds are fixed where the original notebook fixed them (`StratifiedKFold(random_state=10)`); model weight initialisation and `sklearn.utils.shuffle` are not separately seeded, matching the notebook, so exact metric values will still vary run to run.
- This repository does not re-run training to regenerate the paper's numbers — see [Results](#results) for the authors' caveat about the dataset size and augmentation.

---

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `FileNotFoundError: EEG dataset directory not found at ...` | Set `--data-dir` / `ADHD_DATA_DIR` to a valid path, or place the dataset under `data/adhd_dataset/` (see [Data](#data)). |
| `FileNotFoundError: ... missing expected subfolder(s)` | Your dataset directory exists but doesn't have all of `ADHD_part1`, `ADHD_part2`, `Control_part1`, `Control_part2`. |
| `ModuleNotFoundError: No module named 'resnets_utils'` (in Colab) | Only relevant to `ADHD.ipynb`; use `adhd_eeg/resnets_utils.py` from this repo, or add it to Colab's path. |
| `ImportError` / version conflicts installing `tensorflow` | Use a fresh virtualenv with Python 3.9–3.11; `tensorflow<2.16` is pinned in `requirements.txt` for broad OS/GPU compatibility — adjust the pin if you need a newer TF/Keras. |
| Training is very slow | Feature extraction (`mne.time_frequency.tfr_morlet`) and model training are both CPU-intensive; a GPU-backed TensorFlow install is recommended for anything beyond a smoke test. |
| `pandasql` install/runtime errors | `pandasql` (used only by `ADHD.ipynb`, not by `adhd_eeg/`) depends on `sqlalchemy`; if it's problematic in your environment, use the `adhd_eeg` scripts instead, which don't depend on it. |

---

## Results

The accompanying *Cognitive Computation* (2024) paper reports the following cross-validated results on the dataset of **61 children with ADHD** and **60 healthy controls**:

| Model | Level | Accuracy | F1-score |
|-------|-------|----------|----------|
| CNN | Epoch / segment | 92.52% | 93.6% |
| Customised ResNet | Epoch / segment | **96.8%** | **97.1%** |
| CNN | Subject | 96.5% | — |
| Customised ResNet | Subject | **98.6%** | — |

The customised ResNet outperforms the CNN on accuracy, precision, recall, and F1-score.

> **Caveat (from the authors):** the data were augmented and the study is based on a single experiment with a relatively small number of children. The reported accuracies are computed over augmented epoch samples derived from this cohort, so broader validation across a larger and more diverse population is needed before clinical conclusions are drawn.

These numbers are quoted from the published paper; this repository does not independently re-verify them, since doing so requires the private dataset and non-trivial GPU training time (see [Reproducibility notes](#reproducibility-notes)).

### Validation caveat: the folds share subjects

The cross-validation splits at the **segment** level. Each subject contributes many segments, so segments from the same child appear in both the training and the test fold of a given split. A model can therefore recognise the *subject* — individual alpha peak, electrode impedance, persistent artefacts — instead of the *condition*.

The subject id was already computed and passed into `cross_validate`; it was stored as `full_data["person"]` and never used for splitting.

How much this matters can be measured without any EEG data. Using this repository's own `build_folds`, on 20 synthetic subjects whose features encode a per-subject signature and whose labels carry **no generalisable signal at all** (true accuracy = chance = 0.500):

| Split strategy | Accuracy | Test subjects also seen in training |
|---|---|---|
| `segment` (as published) | **1.000** | 62 |
| `subject` | 0.574 | 0 |

The segment strategy reaches perfect accuracy on data containing nothing to learn.

This does **not** show the published 98.6% is entirely leakage — real EEG plausibly carries genuine ADHD signal, and the paper's own caveat about cohort size stands. It does show that the reported figure cannot be read as *subject-level generalisation*: the protocol cannot distinguish learning the condition from recognising the child. The table above also labels one row "Subject", but the published code aggregated votes per **segment**, not per subject.

`cross_validate` now takes two explicit arguments:

```python
# Reproduces the published pipeline (default, leaks subjects)
cross_validate(..., split_by="segment", aggregate_by="segment")

# Leakage-free: no subject spans the split, votes aggregated per child
cross_validate(..., split_by="subject", aggregate_by="person")
```

The default is unchanged so previously reported numbers stay reproducible. Any **new** result should use `split_by="subject"`. Producing a corrected figure requires the private dataset and GPU training; it has not been run here. `tests/test_cv_splits.py` asserts that the leak exists under `segment` and is absent under `subject`.

---

## Tech stack

**Python · TensorFlow / Keras · MNE-Python · scikit-learn · NumPy / SciPy / pandas · Matplotlib · Google Colab**

---

## Citation

If you use this work, please cite the associated paper:

> *Efficient Deep Learning Approach for Diagnosis of ADHD in Children Based on EEG Signals.* Cognitive Computation, 2024.

## License

This repository's code is available under the [MIT License](LICENSE). The EEG dataset itself is not included and may be subject to its own usage terms — check with the original data source before use.

---

*Maintained by [Hamid Jahani](https://github.com/sheperd007).*
