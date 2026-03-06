# RealTaxi Anomaly Detector

## Overview

This project builds a real-time anomaly detection system for NYC taxi activity data.
The goal is to detect sudden drops, spikes, or unusual behaviour in city-level taxi demand
using a comprehensive suite of models ranging from lightweight traditional baselines to
deep temporal architectures including a Transformer encoder.

Target venue: **ICLR Workshop**

## Dataset

We use the NYC Taxi dataset from the [Numenta Anomaly Benchmark (NAB)](https://github.com/zjiaqi725/METER/blob/main/datasets/nyc_taxi.csv).
The data is a 30-minute aggregated time series of passenger counts with binary anomaly labels.
The dataset contains 10,320 time steps with approximately 10% labelled anomalies corresponding
to real-world disruptions (NYC Marathon, Thanksgiving, Christmas, New Year's Day, major snowstorm).

Preprocessed columns:

| Column          | Description                                    |
|-----------------|------------------------------------------------|
| `timestamp`     | Time index of each 30-minute interval          |
| `value`         | Passenger count (model input signal)           |
| `anomaly_score` | Model-predicted anomaly score (inference only) |
| `label`         | Binary ground truth (0 = normal, 1 = anomaly)  |

## Objective

The model generates an anomaly score at each time step and flags abnormal events.
Performance is evaluated using **AUC-ROC**, **AUPRC**, and **detection delay**.

## Models

| Model           | Family          | Description                                                  |
|-----------------|-----------------|--------------------------------------------------------------|
| LOF             | Traditional     | Local Outlier Factor density-based detector                  |
| IsoForest       | Traditional     | Isolation Forest ensemble of random trees                    |
| KNN             | Traditional     | k-Nearest Neighbours distance-based scorer                   |
| Incremental     | Incremental     | Online EMA z-score detector with adaptive statistics         |
| GRU             | Deep Temporal   | Gated Recurrent Unit binary classifier                       |
| LSTM            | Deep Temporal   | Long Short-Term Memory binary classifier                     |
| CNN             | Deep Temporal   | 1-D Convolutional Neural Network binary classifier           |
| Transformer     | Deep Temporal   | Multi-head self-attention encoder classifier                 |
| VAE             | Reconstruction  | Variational Autoencoder reconstruction-error scorer          |
| ENS             | Ensemble        | Mean ensemble of all normalised model scores                 |

## Pipeline

Run from the project root in order:

```bash
python -m scripts.01_make_splits
python -m scripts.02_train_models
python -m scripts.03_generate_scores
python -m scripts.04_select_thresholds
python -m scripts.05_eval_and_plots
```

## Evaluation

Metrics reported for each model:

| Metric           | Description                                                    |
|------------------|----------------------------------------------------------------|
| AUROC            | Area under the ROC curve (ranking ability of anomaly scores)   |
| AUPRC            | Area under the precision-recall curve (class-imbalance robust) |
| Detection Delay  | Mean steps between anomaly onset and first correct detection   |

Results are saved in:

```
outputs/scores.csv          - per-step anomaly scores (test split)
outputs/preds.csv           - binary predictions after threshold selection
outputs/taus.json           - optimal thresholds per model (val F1)
outputs/tables/table1.csv   - AUROC / AUPRC / detection delay table
outputs/figures/            - score distribution, ROC, PR, time-series plots
```

## Project Structure

```
src/
  config.py                 - hyperparameters, paths, data config
  data_utils.py             - data loading, windowing, chronological split
  metrics.py                - AUROC, AUPRC, detection delay, threshold selection
  plotting.py               - score distribution, ROC, PR curve plots
  models/
    lof.py                  - LOF detector
    traditional.py          - Isolation Forest, KNN detectors  [NEW]
    incremental.py          - Online EMA z-score detector       [NEW]
    rnn.py                  - GRU classifier
    lstm.py                 - LSTM classifier                   [NEW]
    cnn.py                  - 1-D CNN classifier                [NEW]
    transformer.py          - Transformer encoder classifier    [NEW]
    vae.py                  - VAE reconstruction scorer
    ensemble.py             - score normalisation and ensemble utilities
  train/
    train_rnn.py            - GRU training loop
    train_vae.py            - VAE training loop
    train_deep.py           - generic deep classifier training loop [NEW]
  infer/
    run_inference.py        - inference utilities
scripts/
  01_make_splits.py         - build chronological train/val/test splits
  02_train_models.py        - train all models                  [UPDATED]
  03_generate_scores.py     - generate anomaly scores           [UPDATED]
  04_select_thresholds.py   - select optimal thresholds         [UPDATED]
  05_eval_and_plots.py      - evaluation table + figures        [UPDATED]
```

## Setup

```bash
pip install -r requirements.txt
```

## References

- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
- Zhou et al. (2025). *Transformer-based anomaly detection for time-series*. https://arxiv.org/abs/2504.04011
- Numenta Anomaly Benchmark (NAB). https://github.com/numenta/NAB
