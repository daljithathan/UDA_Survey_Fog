# UDA for Object Detection in Adverse Weather - Illustrative Experiments

This repository contains the code, configurations, training logs and figures for the illustrative experiments (Figs. 1 and 2) in:

> D. Singh, A. P. S. Chauhan, A. Singla, *Unsupervised Domain Adaptation for Object Detection in Adverse Weather: A Review with Emphasis on Fog and Haze* (under review).

The experiments quantify the domain gap that unsupervised domain adaptation (UDA) aims to close: a YOLOv8s detector trained on clear Cityscapes images is evaluated on Foggy Cityscapes (β = 0.02) and compared with a fully supervised oracle trained on the foggy images.

## Results (Foggy Cityscapes / Cityscapes validation, 500 images)

| Model | Trained on | Evaluated on | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Source-only | Cityscapes (clear) | Cityscapes (clear) | 54.3 | 32.4 |
| Source-only | Cityscapes (clear) | Foggy Cityscapes (β = 0.02) | **29.4** | 18.4 |
| Oracle | Foggy Cityscapes (β = 0.02) | Foggy Cityscapes (β = 0.02) | **49.2** | 31.2 |

Per-class AP@0.5 (%):

| Model / eval set | person | rider | car | truck | bus | train | motorcycle | bicycle | mAP |
|---|---|---|---|---|---|---|---|---|---|
| Source-only / clear | 54.5 | 58.1 | 73.8 | 37.5 | 66.7 | 57.2 | 40.5 | 46.1 | 54.3 |
| Source-only / foggy | 38.8 | 43.6 | 46.3 | 11.8 | 28.6 | 17.1 | 15.4 | 33.2 | 29.4 |
| Oracle / foggy | 53.1 | 54.7 | 71.7 | 33.8 | 57.3 | 44.2 | 35.5 | 43.2 | 49.2 |

## Setup

- Detector: YOLOv8s, initialized from COCO-pretrained weights
- Training: 100 epochs, 640 × 640 input, batch size 32, seed 0, deterministic mode, otherwise default settings
- Evaluation: final-epoch weights (`last.pt`), mAP@0.5
- Software: Ultralytics 8.4.174, PyTorch 2.14.1 (CUDA 13.0); see `requirements.txt`
- Hardware: single NVIDIA H100 NVL GPU

## Data

The datasets are **not** redistributed here, in accordance with the Cityscapes license. Download Cityscapes (`leftImg8bit`, `gtFine`) and Foggy Cityscapes (`leftImg8bit_foggy`) from the official site: https://www.cityscapes-dataset.com

We use the eight standard detection classes (person, rider, car, truck, bus, train, motorcycle, bicycle) in YOLO format, with bounding boxes derived from the `gtFine` instance annotations. The foggy set uses only the β = 0.02 images with the same labels as the clear set (2975 training and 500 validation images each). Edit the `path:` entries in `configs/*.yaml` to point to your local copies.

## Reproduce

```bash
pip install -r requirements.txt
# training (edit the dataset paths in scripts/train_paper.py first)
python scripts/train_paper.py source_only
python scripts/train_paper.py oracle
# evaluation on Foggy Cityscapes (beta = 0.02)
yolo detect val model=runs_paper/source_only/weights/last.pt data=configs/foggy_cityscapes_beta002.yaml imgsz=640 batch=32
yolo detect val model=runs_paper/oracle/weights/last.pt      data=configs/foggy_cityscapes_beta002.yaml imgsz=640 batch=32
# figures
python scripts/make_fig2.py
python scripts/make_fig1.py
```

## Repository contents

- `scripts/`: training, figure-generation and example-selection scripts
- `configs/`: dataset configuration files
- `results/`: training arguments (`args.yaml`), per-epoch metrics (`results.csv`) and full training logs
- `figures/`: Figs. 1 and 2 of the paper (PDF and PNG)
