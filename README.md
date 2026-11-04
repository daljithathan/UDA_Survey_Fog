# UDA for Object Detection in Adverse Weather

This repository holds the code, configurations, logs, results and figures for the experiments in our review paper

> D. Singh, A. P. S. Chauhan, A. Singla. *Unsupervised Domain Adaptation for Object Detection in Adverse Weather. A Review with Emphasis on Fog and Haze* (under review).

There are two sets of experiments. The first one (Figs. 1 and 2 of the paper) shows how much a YOLOv8s detector trained on clear images loses under fog. The second one (Section 4.5, Table 10 and Fig. 8) trains two detectors under one protocol with three seeds and tests them on synthetic fog and on real fog. We wanted to see how much of the progress measured on synthetic fog also shows up on real fog.

## Main finding

Training on labeled synthetic fog raises mAP@0.5 by more than 21 points on Foggy Cityscapes for both detectors. On real fog (Foggy Driving and RTTS) the same training gives no gain beyond the variation between seeds. Simple dehazing with the dark channel prior at test time helps more on real haze than this training does.

## Results

mAP@0.5 (%), mean ± standard deviation over three seeds. FCS is Foggy Cityscapes-DBF and FD is Foggy Driving.

| Detector | Setting | Training or adaptation data | Clear | FCS 0.005 | FCS 0.01 | FCS 0.02 | FD test | FD all | RTTS |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv8s | Source-only | Clear Cityscapes | 53.0±1.1 | 47.3±1.4 | 39.6±0.9 | 29.1±0.3 | 16.3±4.5 | 19.6±2.6 | 21.3±0.4 |
| YOLOv8s | Restoration-based | DCP at test time | 50.8±1.7 | 49.2±1.6 | 47.4±1.3 | 43.8±0.8 | 12.1±0.8 | 18.4±1.0 | 24.2±0.4 |
| YOLOv8s | Source-free | BN statistics on unlabeled FCS 0.02 | 37.6±2.2 | 40.5±1.0 | 36.0±0.8 | 26.4±0.9 | 10.9±2.1 | 15.1±2.0 | 15.8±0.9 |
| YOLOv8s | Synthetic-target supervision | Labeled FCS 0.02 | 38.6±0.7 | 46.0±0.4 | 49.1±0.8 | 50.5±1.2 | 14.5±1.4 | 20.3±1.6 | 22.3±1.1 |
| YOLOv8s | Synthetic-target supervision | Labeled FCS, all densities | 45.9±0.4 | 47.9±0.9 | 47.7±1.1 | 47.3±1.1 | 15.4±1.5 | 19.2±1.9 | 22.6±1.7 |
| RT-DETR-L | Source-only | Clear Cityscapes | 59.1±0.9 | 52.3±0.3 | 44.8±1.2 | 36.0±1.1 | 25.8±3.2 | 28.1±1.9 | 27.4±0.2 |
| RT-DETR-L | Restoration-based | DCP at test time | 58.4±1.3 | 55.9±0.7 | 52.5±0.7 | 46.8±2.0 | 21.3±1.3 | 25.4±0.7 | 29.7±0.3 |
| RT-DETR-L | Source-free | BN statistics on unlabeled FCS 0.02 | 42.6±0.9 | 43.5±1.4 | 39.6±1.1 | 32.6±0.8 | 22.5±2.3 | 25.1±0.6 | 20.7±1.3 |
| RT-DETR-L | Synthetic-target supervision | Labeled FCS 0.02 | 47.0±1.2 | 54.5±1.0 | 57.6±0.6 | 58.5±0.4 | 19.5±2.8 | 25.9±1.7 | 25.7±1.3 |
| RT-DETR-L | Synthetic-target supervision | Labeled FCS, all densities | 50.3±2.9 | 51.7±3.1 | 51.9±3.3 | 50.9±3.9 | 23.2±1.8 | 30.1±1.7 | 27.8±0.6 |

The model trained on labeled Foggy Cityscapes is the oracle on Foggy Cityscapes. On real fog it has never seen a real target image, so there it counts as synthetic-target supervision. FD test has 33 images with fine annotations and FD all has all 101 images. The 33-image split varies a lot between seeds, so the paper uses FD all in the discussion. RTTS has 4321 images and is scored over the five classes it shares with Cityscapes (person, car, bus, motorcycle and bicycle).

The seed 0 run of the YOLOv8s source-only model is the model in Figs. 1 and 2. It gets 54.3% on clear Cityscapes and 29.4% on Foggy Cityscapes at β = 0.02, and the matching oracle gets 49.2%. Per-class results for this run are below.

| Model and test set | person | rider | car | truck | bus | train | motorcycle | bicycle | mAP |
|---|---|---|---|---|---|---|---|---|---|
| Source-only on clear | 54.5 | 58.1 | 73.8 | 37.5 | 66.7 | 57.2 | 40.5 | 46.1 | 54.3 |
| Source-only on fog | 38.8 | 43.6 | 46.3 | 11.8 | 28.6 | 17.1 | 15.4 | 33.2 | 29.4 |
| Oracle on fog | 53.1 | 54.7 | 71.7 | 33.8 | 57.3 | 44.2 | 35.5 | 43.2 | 49.2 |

All raw numbers for every run and test set are in `results/eval_table.csv`.

## Setup

- Detectors are YOLOv8s and RT-DETR-L from Ultralytics, both started from COCO-pretrained weights
- Each model is trained for 100 epochs with 640 × 640 input, batch size 32 for YOLOv8s and 16 for RT-DETR-L, deterministic mode and otherwise default settings
- Every configuration is trained with seeds 0, 1 and 2
- Evaluation uses the final-epoch weights (`last.pt`) and reports mAP@0.5
- BN adaptation recomputes the batch-normalization statistics of the source-only model on the unlabeled Foggy Cityscapes-DBF training images at β = 0.02, with the same letterbox resizing as in evaluation, and trains no weights
- DCP dehazing follows He et al. (2011) with ω = 0.95, t0 = 0.1, a 15 × 15 patch and guided-filter refinement of the transmission (He et al., 2013)
- Software is Ultralytics 8.4.174, PyTorch 2.12.1 with CUDA 13.0 and Python 3.14, see `requirements.txt`
- Hardware is a single NVIDIA H100 NVL GPU

RT-DETR-L uses deformable attention, and its backward pass has no deterministic GPU version. Its runs are therefore not exactly repeatable, even with a fixed seed.

## Data

We do not share the datasets here, in line with their licenses. Please download them from the original sources.

- Cityscapes (`leftImg8bit`, `gtFine`) and Foggy Cityscapes-DBF (`leftImg8bit_foggyDBF`) from https://www.cityscapes-dataset.com
- Foggy Driving from the project page of Sakaridis et al. at people.ee.ethz.ch/~csakarid/SFSU_synthetic
- RTTS from the RESIDE benchmark of Li et al.

We use the eight standard Cityscapes detection classes (person, rider, car, truck, bus, train, motorcycle and bicycle) in YOLO format, with boxes taken from the `gtFine` instance annotations. The three fog densities share the labels of the clear images. Foggy Driving and RTTS labels are mapped to the same class order. The scripts use absolute paths from our server, so please change the paths at the top of each script and in `configs/*.yaml` before running them.

## Reproduce

```bash
pip install -r requirements.txt

# prepare data
python scripts/make_fcs_betas.py      # Foggy Cityscapes-DBF at three densities, YOLO format
python scripts/convert_fd.py          # Foggy Driving to YOLO format with Cityscapes class ids

# train (name, model, data yaml, batch, seed)
python scripts/train_run.py yolov8s_source_s0   yolov8s.pt  configs/cityscapes_clear.yaml 32 0
python scripts/train_run.py yolov8s_oracle_s0   yolov8s.pt  configs/foggy_cityscapes_beta002.yaml 32 0
python scripts/train_run.py yolov8s_fog_all_s0  yolov8s.pt  configs/fcs_all.yaml 32 0
python scripts/train_run.py rtdetr_source_s0    rtdetr-l.pt configs/cityscapes_clear.yaml 16 0
# repeat with seeds 1 and 2, and for the other RT-DETR-L settings

# evaluate a model on all seven test sets
python scripts/eval_all.py yolov8s_source_s0 runs_paper/yolov8s_source_s0/weights/last.pt

# BN adaptation, then evaluation
python scripts/bn_adapt.py runs_paper/yolov8s_source_s0/weights/last.pt <fcs_b0.02/images/train> yolov8s_source_bn_s0.pt
python scripts/eval_all.py yolov8s_source_bn_s0 yolov8s_source_bn_s0.pt

# DCP dehazing of all test sets, then evaluation on the dehazed copies
python scripts/dcp_build.py
python scripts/eval_all.py yolov8s_source_dcp_s0 runs_paper/yolov8s_source_s0/weights/last.pt dcp

# table with mean and standard deviation, and the figures
python scripts/summarize.py
python scripts/fig8.py
python scripts/make_fig2.py
python scripts/make_fig1.py
```

## Repository contents

- `scripts/` holds the data preparation, training, evaluation, adaptation, dehazing and figure scripts
- `configs/` holds the dataset configuration files
- `results/` holds `eval_table.csv` with every result, the training arguments (`args.yaml`), per-epoch metrics (`results.csv`) and the training logs
- `figures/` holds Figs. 1, 2 and 8 of the paper as PDF and PNG

## Citation

If you use this code, please cite the paper above. The citation details will be added once the paper is published.
