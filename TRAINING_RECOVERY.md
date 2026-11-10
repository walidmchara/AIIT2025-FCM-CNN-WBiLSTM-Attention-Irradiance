# Persistent training checkpoints and resume

`training_recovery.py` records an atomic **latest.pt** after every completed
epoch. Model and optimizer weights, RNG states, best validation weights, early
stopping history, and scheduler/AMP state where used are retained.
Checkpoint identity includes experiment metadata so incompatible restarts fail.

## First run

```bash
python train.py --csv /path/to/verified-PVGIS.csv --epochs 50 --out outputs/experiment --checkpoint-dir /content/drive/MyDrive/recovery/AIIT2025-FCM-CNN-WBiLSTM-Attention-Irradiance/experiment_01
```

## Resume after Colab disconnect

Rerun exactly the same command and arguments, adding `--resume`:

```bash
python train.py --csv /path/to/verified-PVGIS.csv --epochs 50 --out outputs/experiment --checkpoint-dir /content/drive/MyDrive/recovery/AIIT2025-FCM-CNN-WBiLSTM-Attention-Irradiance/experiment_01 --resume
```

Replace example data paths with real accessible data. The Drive folder must
be mounted in Colab **before** running the command. On RunPod, replace the
path with the persistent Pod volume. A checkpoint is not guaranteed to survive
if it is stored only in the temporary Colab filesystem.

**Run-specific note:** Physical preprocessing and validation must remain unchanged across resume.

### Boundaries

- Resume starts from the *last fully completed* epoch, rerunning any unfinished epoch.
- Do not change datasets, splits, seed, total epochs, architecture or schedule mid-run.
- Cross-CUDA and cross-platform bitwise equivalence is not guaranteed.
- Checkpoint files can contain executable pickle data; load only trusted files.
- No scientific claims, architecture equations or prior experiment values were changed.
- Checkpoints are intentionally stored outside normal Git history. The automated
  GitHub results publisher currently applies to SG-PIWT, not every repository.
