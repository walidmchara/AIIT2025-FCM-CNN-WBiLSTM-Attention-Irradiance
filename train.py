"""Time-ordered training and evaluation for the FZ–CNN–WBiLSTM–AM model."""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from fz_cnn_wbilstm.data import make_windows, demo_pvgis
from fz_cnn_wbilstm.metrics import regression_metrics
from fz_cnn_wbilstm.model import FuzzyCNNWaveletBiLSTMAttention


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))


def main():
    ap = argparse.ArgumentParser(description='Fuzzy CNN–Morlet BiLSTM–attention experiment')
    ap.add_argument('--csv', type=Path, help='Chronological PVGIS-derived CSV')
    ap.add_argument('--smoke', action='store_true', help='Synthetic pipeline check only')
    ap.add_argument('--target', default='GI')
    ap.add_argument('--time-col', default='time')
    ap.add_argument('--features', default='GI,SE,AT,H,WV', help='Comma-separated historical features')
    ap.add_argument('--config', type=Path, default=ROOT / 'configs' / 'paper.json')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs' / 'latest')
    args = ap.parse_args()
    if args.smoke == bool(args.csv):
        ap.error('Select exactly one: --csv PATH or --smoke')
    cfg = json.loads(args.config.read_text())
    seed_all(cfg['seed'])
    if args.smoke:
        df = demo_pvgis(n=400, seed=cfg['seed'])
        # Reduced width keeps this check fast; paper.json retains published widths.
        cfg = dict(cfg)
        cfg['cnn_filters'] = [8, 16, 24]
        cfg['lstm_units'] = 16
        cfg['attention_units'] = 24
        cfg['batch_size'] = 32
        cfg['epochs'] = 2
    else:
        df = pd.read_csv(args.csv)
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if cfg['epochs'] < 1:
        raise ValueError('epochs must be >= 1')

    features = [s.strip() for s in args.features.split(',') if s.strip()]
    if args.target not in features:
        features.insert(0, args.target)
    dataset = make_windows(df, features=features, target=args.target,
                           lookback=cfg['lookback'], train_fraction=cfg['train_fraction'],
                           val_share_train=cfg['val_share_train'], n_clusters=cfg['n_clusters'],
                           seed=cfg['seed'], time_col=args.time_col)
    net = FuzzyCNNWaveletBiLSTMAttention(
        input_dim=dataset.metadata['features_total'], conv_filters=cfg['cnn_filters'],
        kernel_sizes=cfg['cnn_kernel_sizes'], pool_sizes=cfg['cnn_pool_sizes'],
        lstm_units=cfg['lstm_units'], attention_units=cfg['attention_units'],
        dropout=cfg['dropout'])
    opt = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'])
    criterion = torch.nn.L1Loss()  # MAE matches Table I.
    batches = DataLoader(TensorDataset(dataset.x_train, dataset.y_train),
                         batch_size=cfg['batch_size'], shuffle=True)
    best_loss = float('inf')
    best_state = None
    history = []
    for epoch in range(cfg['epochs']):
        net.train()
        accum = 0.0
        count = 0
        for x, y in batches:
            opt.zero_grad()
            pred = net(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            accum += loss.item() * len(x)
            count += len(x)
        net.eval()
        with torch.no_grad():
            val_loss = float(criterion(net(dataset.x_val), dataset.y_val))
        history.append({'epoch': epoch + 1, 'train_mae_normalized': accum / count,
                        'val_mae_normalized': val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"epoch {epoch + 1}/{cfg['epochs']}: train_mae={accum/count:.5f}, val_mae={val_loss:.5f}")
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        predictions = dataset.inverse_y(net(dataset.x_test).numpy())
        expected = dataset.inverse_y(dataset.y_test.numpy())
    report = {
        'run_type': 'synthetic implementation check' if args.smoke else 'measured-data experiment',
        'publication_doi': cfg['doi'],
        'metrics': regression_metrics(expected, predictions, mape_floor=cfg['mape_floor_W_m2']),
        'config': cfg,
        'data_processing': dataset.metadata,
        'scientific_scope': 'New experiment; published accuracy not independently reproduced',
    }
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': best_state, 'config': cfg,
                'data_processing': dataset.metadata}, args.out / 'checkpoint.pt')
    pd.DataFrame(history).to_csv(args.out / 'training_history.csv', index=False)
    pd.DataFrame({'time': dataset.test_time, 'observed_GI_W_m2': expected,
                  'predicted_GI_W_m2': predictions}).to_csv(args.out / 'test_predictions.csv', index=False)
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
