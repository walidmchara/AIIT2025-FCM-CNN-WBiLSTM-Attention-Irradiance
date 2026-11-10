import numpy as np
import pandas as pd
import pytest
from fz_cnn_wbilstm.data import FuzzyCMeans, make_windows, demo_pvgis


def test_fcm_membership_sums_to_one():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(100, 5))
    model = FuzzyCMeans(3, max_iter=200).fit(x)
    membership = model.predict_membership(x)
    assert membership.shape == (100, 3)
    assert np.allclose(membership.sum(axis=1), 1)
    assert np.isfinite(membership).all()


def test_no_future_targets_and_ordered_splits():
    df = demo_pvgis(400)
    split = make_windows(df, ['GI', 'SE', 'AT', 'H', 'WV'])
    assert len(split.x_train) > len(split.x_val) > 0
    assert len(split.x_test) > 0
    assert split.x_train.shape[1:] == (96, 8)
    assert split.metadata['training_end_row_exclusive'] < split.metadata['validation_end_row_exclusive']
    # The label for t is never included in the t-96:t window.
    original = df.sort_values('time').reset_index(drop=True)
    t0 = split.metadata['test_start_row']
    gi_last_observed = original.loc[t0 - 1, 'GI']
    scaled_last = (gi_last_observed - split.metadata['min_train'][0]) / split.metadata['range_train'][0]
    assert split.x_test[0, -1, 0].item() == pytest.approx(scaled_last, abs=1e-5)


def test_train_only_normalization_no_leakage():
    df = demo_pvgis(400)
    baseline = make_windows(df, ['GI', 'SE', 'AT', 'H', 'WV'])
    modified = df.copy()
    modified.loc[360:, 'GI'] += 1000000
    future = make_windows(modified, ['GI', 'SE', 'AT', 'H', 'WV'])
    assert baseline.metadata['min_train'] == future.metadata['min_train']
    assert baseline.metadata['range_train'] == future.metadata['range_train']
    assert np.allclose(baseline.metadata['fcm_centroids'], future.metadata['fcm_centroids'])


def test_missing_columns_rejected():
    df = demo_pvgis(400)
    with pytest.raises(ValueError, match='Missing dataset columns'):
        make_windows(df.drop(columns=['WV']), ['GI', 'SE', 'AT', 'H', 'WV'])


def test_duplicate_time_rejected():
    df = demo_pvgis(400)
    df.loc[5, 'time'] = df.loc[4, 'time']
    with pytest.raises(ValueError, match='unique'):
        make_windows(df, ['GI', 'SE', 'AT', 'H', 'WV'])
