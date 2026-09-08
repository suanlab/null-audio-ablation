#!/bin/bash
# P1-A: build the isolated env + fetch video-SALMONN 2+ 7B weights & repo.
# CPU/network only — NO GPU use (safe to run while the P1-C sweep holds GPU2).
# Fail-soft: each step logs its rc and we continue, so a flash-attn build
# failure (the known-risky step) does not waste the ~17GB weight download.
set +e
LOG=/tmp/vs2_setup.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date +%T)] VS2 ENV SETUP START"

CONDA=$HOME/miniconda3/bin/conda
ENVP=$CONDA_ENVS/videosalmonn2
PIP=$ENVP/bin/pip
PY=$ENVP/bin/python
REPO=$WORK_DIR/video-SALMONN-2
WEIGHTS=$MODELS_DIR/video-SALMONN2_plus_7B_full
declare -A RC

echo "[$(date +%T)] STEP git clone repo"
git clone --depth 1 https://github.com/bytedance/video-SALMONN-2 "$REPO"; RC[clone]=$?

echo "[$(date +%T)] STEP conda create"
"$CONDA" create -y -n videosalmonn2 python=3.10; RC[conda]=$?

echo "[$(date +%T)] STEP pip download helpers"
"$PIP" install -q huggingface_hub soundfile librosa; RC[helpers]=$?

echo "[$(date +%T)] STEP weight snapshot_download (~17GB, long)"
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="tsinghua-ee/video-SALMONN2_plus_7B_full",
    local_dir="$MODELS_DIR/video-SALMONN2_plus_7B_full",
    local_dir_use_symlinks=False,
)
print("WEIGHTS_AT", p)
PY
RC[weights]=$?

echo "[$(date +%T)] STEP pip core stack"
"$PIP" install -q torch==2.7.1 torchaudio==2.5.1 torchvision==0.22.1 torchcodec==0.4.0; RC[torch]=$?
"$PIP" install -q transformers==4.51.3 tokenizers==0.21.0 accelerate==1.7.0 peft==0.15.2 \
  deepspeed==0.16.0 triton==3.3.1 numpy==1.24.4 decord==0.6.0 liger_kernel==0.5.10; RC[stack]=$?

echo "[$(date +%T)] STEP flash-attn (RISKY, non-fatal)"
"$PIP" install -q flash_attn==2.7.4.post1 --no-build-isolation; RC[flash]=$?
[ "${RC[flash]}" -ne 0 ] && echo "[$(date +%T)] FLASH_ATTN_FAILED rc=${RC[flash]} (fallback: original-7B older stack)"

echo "[$(date +%T)] RC SUMMARY:"
for k in clone conda helpers weights torch stack flash; do echo "  $k=${RC[$k]:-NA}"; done
FAIL=0
for k in clone conda helpers weights torch stack; do [ "${RC[$k]:-1}" -ne 0 ] && FAIL=1; done
if [ "$FAIL" -eq 0 ]; then echo "[$(date +%T)] VS2 ENV SETUP DONE OK (flash=${RC[flash]:-NA})";
else echo "[$(date +%T)] VS2 ENV SETUP DONE WITH FAILURES (see RC SUMMARY)"; fi
