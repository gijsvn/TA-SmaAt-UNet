# TA-SmaAt-UNet

Code for the paper **"THIS IS A PLACEHOLDER"** by Gijs van Nieuwkoop and Siamak Mehrkanoon.

This repository implements **TA-SmaAt-UNet**, a time-aware extension of SmaAt-UNet for radar-based precipitation nowcasting. The model uses the recent precipitation sequence together with a compact temporal feature vector containing cyclical encodings of time-of-day and time-of-year. These features are injected through lightweight temporal conditioning layers in the SmaAt-UNet encoder-decoder architecture.

The main model can be found in:

```text
models/TA_SmaAt_UNet/
```

<p align="center">
  <img src="assets/TA-SmaAt-UNet topology.png" width="850">
</p>

<p align="center">
  <em>TA-SmaAt-UNet extends SmaAt-UNet with lightweight temporal conditioning layers based on time-of-day and time-of-year encodings.</em>
</p>

## Overview

The default nowcasting task is:

```text
18 input radar frames  ->  12 future radar frames
90 minutes of history  ->  5 to 60 minutes ahead
5-minute time interval
```

The repository contains scripts for training, evaluation, persistence baselines, model comparison, and the layer-conductance analysis used in the paper.

| Script | Description |
| --- | --- |
| `train.py` | Train `SmaAt-UNet` or `TA-SmaAt-UNet`. |
| `eval.py` | Evaluate a trained model. |
| `eval_persistence.py` | Evaluate the persistence baseline. |
| `compare_models.py` | Compare evaluated models and create plots. |
| `conductance_analysis.py` | Run the TA-SmaAt-UNet layer-conductance analysis. |

## Repository structure

```text
.
├── models/
│   ├── SmaAt_UNet/            # SmaAt-UNet implementation
│   ├── TA_SmaAt_UNet/         # Proposed time-aware model
│   └── lightning_base.py      # PyTorch Lightning wrapper
├── util/
│   ├── callbacks.py
│   ├── evaluate_model.py
│   ├── load_dataset.py
│   └── visualization.py
├── train.py
├── eval.py
├── eval_persistence.py
├── compare_models.py
└── conductance_analysis.py
```

## Installing dependencies

Create a Python environment and install the required packages:

```bash
git clone <repository-url>
cd <repository-name>

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

For GPU training, install a PyTorch build matching your CUDA version. The scripts automatically use CUDA when available.

## Data

The dataset is not included in this repository. The scripts expect an HDF5 file readable by `util/load_dataset.py`.

Expected structure:

```text
precipitation_dataset.h5
├── train/
│   ├── images       # [N_train, n_input_imgs + n_output_imgs, H, W]
│   └── timestamps   # [N_train, n_input_imgs + n_output_imgs, 1]
└── test/
    ├── images       # [N_test, n_input_imgs + n_output_imgs, H, W]
    └── timestamps   # [N_test, n_input_imgs + n_output_imgs, 1]
```

For the default setup, each sample contains 30 frames:

```text
images: [N, 30, H, W]
```

The first 18 frames are used as input and the next 12 frames are used as the target. The timestamp of the final frame in the sequence is converted into the 4-dimensional temporal vector used by TA-SmaAt-UNet.

Each dataset item has the form:

```python
((x, t), y)
```

where:

```text
x: [18, H, W]     input precipitation maps
t: [4]            temporal context vector
y: [12, H, W]     target precipitation maps
```

Validation data is sampled from the `train` group by `PrecipitationDataModule`; the `test` group is used only for testing. The default validation fraction is 0.1 and the split is stratified over the annual cycle using the temporal vectors.

In the paper, the experiments use KNMI radar precipitation maps over the Netherlands from 2016 to 2019, with 2016-2018 used for training and 2019 used for testing.

## Training

Train the proposed model:

```bash
python train.py \
  --model TA-SmaAt-UNet \
  --data-file data/precipitation_dataset.h5 \
  --result-log-dir results
```

Train the SmaAt-UNet baseline:

```bash
python train.py \
  --model SmaAt-UNet \
  --data-file data/precipitation_dataset.h5 \
  --result-log-dir results
```

Important default training settings are:

```text
batch size:       16
loss:             MSE
learning rate:    1e-3
early stopping:   15 epochs without validation improvement
LR scheduler:     reduce on validation plateau, patience 4
```

These can be changed from the command line. For example:

```bash
python train.py \
  --model TA-SmaAt-UNet \
  --data-file data/precipitation_dataset.h5 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --max-epochs 300 \
  --loss mse \
  --seed 42
```

Each run saves logs, checkpoints, visualizations, and a `config.json` file in `results/`.

## Evaluation

Evaluate a trained model with:

```bash
python eval.py \
  --checkpoint-path results/comparison/ta_smaat_unet/model.ckpt \
  --data-file data/precipitation_dataset.h5 \
  --evaluation-set test \
  --precipitation-thresholds 0.5 10.0 20.0
```

The checkpoint directory must also contain the corresponding `config.json` file. By default, the script writes:

```text
test_results.json
test_results_seasonal.json
test_predictions.png
```

The evaluation metrics include MSE and threshold-based precipitation metrics such as CSI, POD, FAR, and MCC.

## Persistence baseline

The persistence baseline repeats the last observed input frame for all future lead times:

```bash
python eval_persistence.py \
  --data-file data/precipitation_dataset.h5 \
  --output-dir results/comparison/persistence \
  --evaluation-set test \
  --precipitation-thresholds 0.5 10.0 20.0
```

## Comparing models

After evaluating each model, place the result folders in one comparison directory, for example:

```text
results/comparison/
├── smaat_unet/
│   ├── model.ckpt
│   ├── config.json
│   └── test_results.json
├── ta_smaat_unet/
│   ├── model.ckpt
│   ├── config.json
│   └── test_results.json
└── persistence/
    └── test_results.json
```

Then run:

```bash
python compare_models.py \
  --model-folder results/comparison \
  --data-file data/precipitation_dataset.h5 \
  --dataset test \
  --thresholds 0.5 10.0 20.0
```

This prints metric tables and generates comparison plots, including CSI across lead times, seasonal CSI, predicted rainfall-intensity distributions, and example nowcasts.

## Layer-conductance analysis

To reproduce the conductance analysis for TA-SmaAt-UNet:

```bash
python conductance_analysis.py \
  --data-file data/precipitation_dataset.h5 \
  --checkpoint results/comparison/ta_smaat_unet/model.ckpt \
  --output-dir results/conductance/ta_smaat_unet_by_lead \
  --evaluation-set test
```

This saves CSV summaries and the conductance figure in the output directory.

## Acknowledgements

This work builds on SmaAt-UNet:

Kevin Trebing, Tomasz Stańczyk, and Siamak Mehrkanoon, **SmaAt-UNet: Precipitation Nowcasting using a Small Attention-UNet Architecture**, *Pattern Recognition Letters*, 2021.

The experiments in the paper use radar precipitation data from the Royal Netherlands Meteorological Institute (KNMI).

## Citation

If you use this repository, please cite the accompanying paper. 

```
PLACEHOLDER
```

Please also cite SmaAt-UNet if you use the underlying architecture:

```bibtex
@article{trebing2021smaat,
  title={SmaAt-UNet: Precipitation nowcasting using a small attention-UNet architecture},
  author={Trebing, Kevin and Sta\'{n}czyk, Tomasz and Mehrkanoon, Siamak},
  journal={Pattern Recognition Letters},
  volume={145},
  pages={178--186},
  year={2021},
  publisher={Elsevier}
}
```
