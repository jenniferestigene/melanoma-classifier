# Melanoma Classifier

A convolutional neural network, built in PyTorch, that
classifies dermoscopic skin lesion images as benign or malignant.

**Test accuracy: 88.6%** &nbsp;|&nbsp; **Malignant recall: 88.8%** — see
[`results.md`](results.md) for training curves, the confusion matrix, and a
record of what was tried.

## Motivation

This project explores image-based classification in a biomedical context, part of a broader interest in applying machine learning to molecular and
biological data, and in scientific tools that move from raw data to usable
signal. Skin lesion classification is a well-studied benchmark task in
medical imaging, which makes it a good testbed for practicing the full
pipeline: data preprocessing, model architecture design, training
diagnostics, and honest evaluation of what a model can and can't
responsibly be used for.

## Dataset

- **Source:** [Melanoma Cancer Image Dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset) (Kaggle)
- **Classes:** benign, malignant (binary classification)
- **Split:** balanced training set (4,605 images per class), 1,000-image
  held-out test set (500 per class)
- **Preprocessing:** images resized to 100×100, RGB, normalized to [0, 1]

The dataset itself is not included in this repository — see
[Setup](#setup) for how to obtain it.

## Architecture

A 3-layer CNN (no pretrained weights):

```
Input (3×100×100)
  → Conv2d(3→32, k=5) → ReLU → MaxPool(2×2)
  → Conv2d(32→64, k=5) → ReLU → MaxPool(2×2)
  → Conv2d(64→128, k=5) → ReLU → MaxPool(2×2)
  → Flatten (128×9×9 = 10,368)
  → Linear(10368→512) → ReLU
  → Linear(512→2)
```

Trained with Adam (lr=0.001), cross-entropy loss, batch size 100, for 20
epochs. Full details in [`model.py`](model.py) and [`train.py`](train.py).

## Results

| Metric | Value |
|---|---|
| Test accuracy | **88.6%** |
| Malignant recall (sensitivity) | **88.8%** |
| Malignant precision | 88.4% |
| Benign specificity | 88.4% |
| Final validation accuracy | 89.4% |

![Training loss and validation accuracy over 20 epochs](assets/training_curves.png)

![Confusion matrix on the test set](assets/confusion_matrix.png)

Of 500 malignant test cases, 56 were missed (classified as benign) and 444
were correctly caught. Full epoch-by-epoch metrics, the raw
[`training_log.csv`](training_log.csv), and a record of transfer-learning
experiments that were tried and did not outperform this model, are in
[`results.md`](results.md).

## Limitations

This model is **not clinically viable** and is not intended for diagnostic
use. 88.6% accuracy, and 88.8% malignant recall specifically, fall well
short of the reliability that medical screening requires as a false
negative (malignant misclassified as benign) is the costly error in this
setting, and 56 such cases occurred in this test run alone. See
[`results.md`](results.md#known-limitations) for a fuller discussion.

## Setup

```bash
git clone https://github.com/jenniferestigene/melanoma-classifier.git
cd melanoma-classifier
python3 -m venv venv
source venv/bin/activate        # <- Mac, Windows -> : venv\Scripts\activate
pip install -r requirements.txt
```

Download the [dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images?resource=download)
and place it in the repo root as `melanoma_cancer_dataset/`, matching this
structure:

```
melanoma_cancer_dataset/
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

## Usage

Run in order:

```bash
python preprocess.py          # builds melanoma_training_data.npy / melanoma_testing_data.npy
python train.py               # trains the model, saves saved_model.pth + training_log.csv
python visualize_training.py  # builds assets/training_curves.png from the training log
python test.py                # evaluates on the test set, builds assets/confusion_matrix.png
```

`train.py` automatically uses a GPU if one is available
(`torch.cuda.is_available()`), and falls back to CPU otherwise, the same
code runs unmodified locally or on a GPU-backed environment like Google
Colab.

## Future Work

- **Data augmentation** (random flips, rotation, color jitter) to improve
  generalization from a relatively small dataset, likely the most direct
  lever for reducing the 56 missed malignant cases seen in the current run
- **Transfer learning**, revisited with a corrected implementation, 
  initial attempts (documented in [`results.md`](results.md)) ran into
  overfitting and BatchNorm-related underfitting; worth retrying with
  proper backbone freezing and/or a staged unfreeze schedule
- **Higher input resolution / color-aware preprocessing** — the ABCDE
  dermatological criteria rely heavily on color and fine border detail,
  which 100×100 downsampling likely loses
- **Bias/fairness analysis** across skin tones and lesion types, which
  this evaluation does not currently cover

## Repository structure

```
melanoma-classifier/
├── preprocess.py           # raw images → .npy tensors
├── model.py                # CNN architecture (nn.Module)
├── train.py                # training loop, writes training_log.csv
├── test.py                 # evaluation + confusion matrix
├── visualize_training.py   # builds training_curves.png from the log
├── requirements.txt
├── training_log.csv        # metrics from the reported run
├── results.md              # training curves, metrics, experiment log
├── assets/
│   ├── training_curves.png
│   └── confusion_matrix.png
└── README.md
```