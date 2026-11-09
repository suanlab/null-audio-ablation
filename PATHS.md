# Path variables used in this repository

The scripts were written against a specific machine. Absolute paths have been replaced with
variables so the procedure stays inspectable without publishing that layout. Set these
before running anything that touches media or checkpoints:

| Variable | Meaning |
|---|---|
| `$REPO_ROOT` | This repository |
| `$WORK_DIR` | Parent directory holding the third-party model checkouts (`video-SALMONN-2/` under it) |
| `$VIDEOLLAMA2_DIR` | A VideoLLaMA2 checkout at the commit in `MODELS.md`, with `external/videollama2/eval_avqa_5mode.py` copied into its root |
| `$VIDEOSALMONN_V1_DIR` | A video-SALMONN v1 checkout (for the BEATs checkpoint the probe uses) |
| `$MODELS_DIR` | Downloaded model weights |
| `$CONDA_ENVS` | Conda environment root; each model needs its own environment (`envs/`) |
| `$VENV` | The analysis environment (`requirements-frozen.txt`) |

Shell scripts expand these as ordinary shell variables. Python scripts read them from the
environment when the path is first needed and exit naming the variable if it is unset, so
`--help` and imports work without them. Scripts that only read `eval_results/` (the figure
and macro generator, `primary_family.py`, `delivery_gain.py`) need none of them.

Media paths are intentionally not provided: no dataset media is redistributed here.
