# FZ–CNN–WBiLSTM–Attention for Global Irradiance Forecasting

**Paper:** *An Advanced Forecasting Model for Global Irradiance in Solar Electric Vehicles Using a Fuzzy C-Means-CNN-Wavelet BiLSTM Network with an Attention Mechanism*  
**Conference:** IEEE International Conference on Innovation in Artificial Intelligence and Internet of Things (AIIT), 2025  
**DOI:** https://doi.org/10.1109/AIIT63112.2025.11082921

## Research method

This project studies a multi-stage irradiance forecasting architecture in which fuzzy c-means describes overlapping meteorological regimes, convolutional layers extract local characteristics, a Morlet-enhanced bidirectional LSTM learns temporal structure, and additive attention aggregates the recurrent outputs into a global irradiance estimate.

The article reports experiments for Riyadh, Medina, Mecca and Dammam using PVGIS measurements from 2005–2020. The experimental input variables comprise irradiance, solar elevation, ambient temperature, humidity and wind speed. The published model configuration, where specified, is summarized in [`configs/paper.json`](configs/paper.json) and [`docs/ARCHITECTURE_AND_VALIDATION.md`](docs/ARCHITECTURE_AND_VALIDATION.md).

## Model specification

| Stage | Configuration |
| --- | --- |
| FCM | Tolerance 0.0001; at most 2,000 iterations |
| CNN | Filters 64 → 128 → 256; kernel 5; max pooling 3 per layer |
| Wavelet recurrent model | BiLSTM with 150 units and real Morlet activation |
| Attention | 300 hidden units, softmax weighting |
| Regularization | Dropout 0.4 |
| Training | Adam, MAE loss |
| Holdout | Chronological 80/20 split |

The model applies Morlet to the recurrent sequence representation. The precise gate-level wavelet implementation remains an open replication detail. The number of fuzzy regimes and some training choices are explicitly configurable rather than attributed to the original paper.

## Environment

```bash
python -m venv .venv
pip install -r requirements.txt
pytest -q
python train.py --smoke --epochs 2 --out outputs/smoke
```

For a measured PVGIS CSV with columns `time,GI,SE,AT,H,WV`:

```bash
python train.py --csv data/riyadh_pvgis.csv --target GI --time-col time --epochs 40 --out outputs/riyadh
```

The workflow writes a model checkpoint, epoch-level training history, timestamped test predictions and JSON evaluation metrics. Each city is evaluated separately; the CSV must preserve chronological order.

## Experiment integrity

The data-processing code learns fuzzy centroids and normalization parameters from training observations only. Regression metrics include MAE, RMSE, MBE, R², rRMSE and daylight-filtered MAPE. The original PVGIS CSVs are not packaged; the smoke-test signal is deliberately synthetic and only checks software execution. Published tables are **not** reproduced or claimed validated by the smoke run.

See [`data/README.md`](data/README.md) for the source-data contract and [`docs/ARCHITECTURE_AND_VALIDATION.md`](docs/ARCHITECTURE_AND_VALIDATION.md) for deviations and verification requirements.
