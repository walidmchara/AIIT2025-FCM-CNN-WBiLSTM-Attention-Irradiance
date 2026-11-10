"""Solar-irradiance metrics with an explicit nighttime MAPE policy."""
import numpy as np


def regression_metrics(y_true, y_pred, mape_floor=10.0) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0 or not np.isfinite(y_pred).all():
        raise ValueError("Invalid prediction and target arrays")
    errors = y_pred - y_true
    mse = float(np.mean(errors ** 2))
    mask = np.abs(y_true) > mape_floor
    return {
        "MAE_W_m2": float(np.mean(np.abs(errors))),
        "MABE_W_m2": float(np.mean(np.abs(errors))),
        "MBE_W_m2": float(np.mean(y_true - y_pred)),
        "RMSE_W_m2": float(np.sqrt(mse)),
        "R2": float(1 - np.sum(errors ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + 1e-12)),
        "rRMSE_pct": float(100 * np.sqrt(mse) / y_true.mean()) if abs(y_true.mean()) > 1e-12 else None,
        "MAPE_daylight_pct": float(100 * np.mean(np.abs(errors[mask]) / np.abs(y_true[mask]))) if mask.any() else None,
        "mape_floor_W_m2": mape_floor,
        "mape_daylight_samples": int(mask.sum()),
    }
