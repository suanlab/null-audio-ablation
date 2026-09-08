# Path variables used in this repository

The scripts were written against a specific machine. Absolute paths have been replaced with
variables so the procedure stays inspectable without publishing that layout. Set these
before running anything that touches media or checkpoints:

| Variable | Meaning |
|---|---|
| `$REPO_ROOT` | This repository |
| `$WORK_DIR` | Parent directory holding the third-party model checkouts |
| `$VIDEOLLAMA2_DIR` | A VideoLLaMA2 checkout (its adapter is patched in place) |
| `$VIDEOSALMONN_V1_DIR` | A video-SALMONN v1 checkout |
| `$MODELS_DIR` | Downloaded model weights |
| `$CONDA_ENVS` | Conda environment root; each model needs its own pinned environment |
| `$VENV` | The analysis environment (`pip install -e ".[dev]"`) |

Media paths are intentionally not provided: no dataset media is redistributed here.
