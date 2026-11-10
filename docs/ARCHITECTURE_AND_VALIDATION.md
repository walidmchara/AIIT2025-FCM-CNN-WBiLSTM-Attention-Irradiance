# Architecture and experimental protocol

## Article-aligned components

- Fuzzy C-means memberships: convergence tolerance `1e-4`, maximum 2,000 iterations (Table I).
- Three 1D CNN blocks: filters `[64, 128, 256]`, kernel sizes `[5, 5, 5]` and max-pool sizes `[3, 3, 3]` (Table I).
- Morlet wavelet function: `cos(1.75*s) exp(-s*s/2)` (Equation 8).
- Bidirectional LSTM: 150 units in each direction (Table I).
- Attention hidden layer: 300 units with softmax weighting over recurrent timesteps (Table I; Equations 9–11).
- Dropout: 0.4; loss: MAE; optimizer: Adam (Table I).
- Time-series development/test fractions: 80% / 20% (Results and Analysis).

The original paper supplies the *Morlet activation formula*, but does not fully detail how this activation is embedded within recurrent cell gates. This implementation applies the wavelet activation to the bidirectional sequence representation. It must not be interpreted as source-identical gate-level WBiLSTM.

## Configurable experiment choices

The published article does not unambiguously fix the number of FCM regimes, historical window length, batch size, epochs, optimizer learning rate, source CSV ordering or validation slice. `configs/paper.json` makes those choices explicit, including three fuzzy regimes, a 96-hour lookback and a separate chronological validation slice. These are not reported as verified original settings.

## Data integrity

All fuzzy centroids and min-max normalization statistics are fitted to development samples. Only historical GI values are present in an input window for the target GI at the next timestamp. No future target is used for feature computation. Cross-city experiments should not use combined normalization across held-out locations.

MAPE is calculated only when the absolute observed GI exceeds the configured 10 W/m² threshold, and the number of included samples is logged. This policy is not asserted as the original article's denominator convention.

`--smoke` intentionally reduces network widths for a fast pipeline test; results obtained on synthetic data do not represent paper figures.
