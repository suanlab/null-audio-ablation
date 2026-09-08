#!/bin/bash
# prep_supplementary.sh — Build an anonymized supplementary code bundle for
# ARR / EMNLP submission. Run this BEFORE uploading to the submission portal.
#
# What it does:
#  - copies the minimal code subset to a fresh ./supplementary_zip/ directory
#  - sed-replaces every $HOME and $WORK_DIR path
#    with placeholder variables so author identity is not leaked
#  - excludes private notes (.omc/, MEMORY.md, CHECK.md, PLAN.md, AGENTS.md,
#    CLAUDE.md, .git/)
#  - excludes large weight/data blobs
#  - zips the result for upload
#
# This script does NOT modify the live repo; it produces a separate
# supplementary_zip/ directory and a tarball.
set -euo pipefail

ROOT=$REPO_ROOT
OUT="$ROOT/supplementary_zip"
TAR="$ROOT/supplementary.tar.gz"

rm -rf "$OUT" "$TAR"
mkdir -p "$OUT"

echo "[$(date +%T)] copying minimal code subset..."
# Source code (core)
cp -r "$ROOT/src" "$OUT/src"
# Scripts (eval + aggregate + audit + run + setup, exclude obviously personal ones)
mkdir -p "$OUT/scripts"
for f in eval_avqa.py eval_avqa_fast.py eval_avqa_qwenomni.py \
         eval_avqa_videosalmonn2.py aggregate_matched_98.py \
         aggregate_shift_sweep.py aggregate_p1b_large.py \
         aggregate_shapley.py aggregate_visionhard.py \
         conditional_taxonomy.py shapley_ci.py three_vs_five_mode.py \
         equiv_tost.py make_dissociation_figure.py \
         make_grounding_spectrum_figure.py \
         audit_claims.py extract_failure_taxonomy.py categorize_avqa.py \
         compute_sample_weights.py run_poscontrol.sh run_shift_sweep.sh \
         run_videosalmonn2.sh run_p1b_large.sh setup_videosalmonn2_env.sh \
         fix_videosalmonn2_torch.sh; do
  [ -f "$ROOT/scripts/$f" ] && cp "$ROOT/scripts/$f" "$OUT/scripts/$f" || true
done
# External adapter (kept in a sibling directory in the live repo)
mkdir -p "$OUT/external_adapters/videollama2"
cp "$ROOT/../VideoLLaMA2/eval_avqa_5mode.py" "$OUT/external_adapters/videollama2/" 2>/dev/null || true
# Configs + tests + dataset index lists
cp -r "$ROOT/configs" "$OUT/configs" 2>/dev/null || true
cp -r "$ROOT/tests" "$OUT/tests"
mkdir -p "$OUT/data/instruct" "$OUT/eval_results/matched_98"
cp "$ROOT/data/instruct/avqa_test_clean.jsonl" "$OUT/data/instruct/" 2>/dev/null || true
for j in audio_explicit_idx.json categories.json failure_taxonomy.json \
         shapley.json visionhard.json conditional_taxonomy.json \
         shapley_ci.json three_vs_five.json tost_shift.json \
         summary_with_ci.json; do
  cp "$ROOT/eval_results/matched_98/$j" "$OUT/eval_results/matched_98/" 2>/dev/null || true
done

# Per-sample prediction dumps — every JSON cited by scripts/audit_claims.py
# (each entry has {index, video, gt, pred, correct, output}). Total ~2.1 MB.
# These let reviewers re-run the full audit pipeline from raw model outputs.
mkdir -p "$OUT/eval_results/shift_sweep"
# Matched-98 cross-model: real/silent/noise/shuffled/shifted (+1s if present)
for prefix in full_v6 videollama2 vsalm2 qwenomni no_bridge vision_only; do
  for mode in real silent noise shuffled shifted shifted_1s; do
    cp "$ROOT/eval_results/${prefix}_${mode}.json" "$OUT/eval_results/" 2>/dev/null || true
  done
done
# Positive control (audio-explicit n=47, blanked video): 4 models × {real, silent}
for prefix in poscontrol_agta poscontrol_videollama2 poscontrol_vsalm2 poscontrol_qwenomni; do
  for mode in real silent; do
    cp "$ROOT/eval_results/${prefix}_${mode}.json" "$OUT/eval_results/" 2>/dev/null || true
  done
done
# Larger-n (n=548) external robustness: 2 models × {real, silent, shifted}
for prefix in large548_videollama2 large548_vsalm2; do
  for mode in real silent shifted; do
    cp "$ROOT/eval_results/${prefix}_${mode}.json" "$OUT/eval_results/" 2>/dev/null || true
  done
done
# Seven-point shift-magnitude sweep (Appendix Table 13): AGTA + VL2
cp "$ROOT/eval_results/shift_sweep/"*.json "$OUT/eval_results/shift_sweep/" 2>/dev/null || true
# Training-recipe ablations (Appendix Table 8): QB v2/v3/v4 + GRPO variants
for prefix in qb_v2 qb_v3 qb_v4 qb_v5_grpo qb_v5b_grpo; do
  for mode in real silent noise shuffled shifted; do
    cp "$ROOT/eval_results/${prefix}_${mode}.json" "$OUT/eval_results/" 2>/dev/null || true
  done
done

# NOTE: LaTeX source (main.tex / references.bib / figure PDFs) is NOT bundled
# here. ARR's "Software" field is intended for software; the PDF itself is the
# canonical artifact. Reviewers can still re-derive every paper number from the
# bundled per-sample prediction dumps via scripts/audit_claims.py.
# Build metadata
cp "$ROOT/requirements-frozen.txt" "$OUT/" 2>/dev/null || true
cp "$ROOT/pyproject.toml" "$OUT/" 2>/dev/null || true

# Anonymize: replace personal paths with placeholder env vars
echo "[$(date +%T)] anonymizing paths (sed)..."
grep -rIl "$HOME\|$WORK_DIR" "$OUT/" 2>/dev/null | while read -r f; do
  sed -i \
    -e 's|$HOME/\.venv/videollm/bin/python|python|g' \
    -e 's|$HOME/miniconda3/envs/\([a-zA-Z0-9_]*\)/bin/python|python|g' \
    -e 's|$HOME/miniconda3/envs/\([a-zA-Z0-9_]*\)/bin/pip|pip|g' \
    -e 's|$HOME/miniconda3/envs/\([a-zA-Z0-9_]*\)|${CONDA_PREFIX}|g' \
    -e 's|$HOME/miniconda3|${CONDA_HOME}|g' \
    -e 's|$HOME/\.venv/\([a-zA-Z0-9_]*\)|${VENV_ROOT}/\1|g' \
    -e 's|$HOME|${HOME}|g' \
    -e 's|$REPO_ROOT|${REPO_ROOT}|g' \
    -e 's|$VIDEOLLAMA2_DIR|${EXT_ROOT}/VideoLLaMA2|g' \
    -e 's|$WORK_DIR/video-SALMONN-2|${EXT_ROOT}/video-SALMONN-2|g' \
    -e 's|$MODELS_DIR|${MODELS_ROOT}|g' \
    -e 's|$WORK_DIR|${WORKSPACE}|g' \
    "$f"
done

# Add an anonymized README for the supplementary
cat > "$OUT/README_supp.md" <<'EOF'
# Anonymous Supplementary Code — Five-Mode Audio-Intervention Audit

This bundle reproduces the paper's headline numbers from on-disk evaluation
artifacts (`eval_results/`) and the four evaluation adapters in `scripts/`
and `external_adapters/`.

## Environment

Three separate Python environments are needed (see `requirements-frozen.txt`
plus `scripts/setup_videosalmonn2_env.sh`). Set the env-var placeholders
below to point at your machine:

```
export HOME=/path/to/your/home
export REPO_ROOT=$HOME/path/to/this/bundle
export EXT_ROOT=$HOME/external_repos
export MODELS_ROOT=$HOME/models
export CONDA_HOME=$HOME/miniconda3
export WORKSPACE=$HOME/workspace
```

## Reproduce the headline audit (1 command, no GPU)

```bash
python scripts/audit_claims.py
```

This re-verifies every numeric claim in the paper against the per-mode JSON
files in `eval_results/` and reports `RESULT: ALL PASS / FINAL: ALL PASS`.

## Reproduce a single eval (requires the corresponding env + GPU)

See `scripts/run_*.sh` for each of:
- `run_shift_sweep.sh`     (Appendix shift-magnitude sweep)
- `run_videosalmonn2.sh`   (video-SALMONN 2+ 5-mode + positive control)
- `run_p1b_large.sh`       (n=548 robustness for the two positive-controlled externals)
- `run_poscontrol.sh`      (blanked-video positive control for AGTA + 2 externals)

Each script pins to a specific GPU (`CUDA_VISIBLE_DEVICES=2` in our setup);
adjust to your hardware.

## Tests

```bash
pytest tests/ -x -q
```

In particular `tests/test_temporal.py` verifies the AGTA bridge zero-init
contract and `tests/test_eval_adapter_shuffled_consistency.py` verifies that
all four eval adapters produce the IDENTICAL shuffled donor mapping at
(n, seed)=(98, 42) — a permanent regression gate for the cross-model
R-Sh comparison.

EOF

# Pack
echo "[$(date +%T)] packing supplementary.tar.gz..."
cd "$ROOT"
tar -czf "$TAR" -C "$OUT" .
echo "[$(date +%T)] anonymized supplementary built:"
echo "  dir: $OUT"
echo "  tar: $TAR ($(du -h "$TAR" | cut -f1))"
echo "Leftover Suan/personal paths to manually inspect (should be empty):"
grep -rIn "Suan\|$HOME\|$WORK_ROOT" "$OUT/" 2>/dev/null | head -10 || true
