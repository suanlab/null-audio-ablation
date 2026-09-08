#!/bin/bash
# P1-A fix: the repo's pinned torchaudio==2.5.1 conflicts with torch==2.7.1.
# Correct compatible trio for torch 2.7.1 is torchaudio 2.7.1 / torchvision
# 0.22.1. Reinstall into the existing videosalmonn2 env, then redo the deps
# that need torch present (torchcodec, liger/deepspeed, flash-attn).
# CPU/network + a CUDA compile — NO GPU runtime use (parallel-safe w/ sweep).
set +e
LOG=/tmp/vs2_fix.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
PIP=$CONDA_ENVS/videosalmonn2/bin/pip
PY=$CONDA_ENVS/videosalmonn2/bin/python
declare -A RC
echo "[$(date +%T)] VS2 TORCH FIX START"

echo "[$(date +%T)] STEP torch trio (corrected: torchaudio 2.7.1)"
"$PIP" install -q torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1; RC[torch]=$?

echo "[$(date +%T)] STEP torchcodec + core stack (needs torch present)"
"$PIP" install -q torchcodec==0.4.0; RC[codec]=$?
"$PIP" install -q transformers==4.51.3 tokenizers==0.21.0 accelerate==1.7.0 peft==0.15.2 \
  deepspeed==0.16.0 triton==3.3.1 numpy==1.24.4 decord==0.6.0 liger_kernel==0.5.10; RC[stack]=$?

echo "[$(date +%T)] STEP flash-attn (RISKY, non-fatal; needs torch present)"
"$PIP" install -q flash_attn==2.7.4.post1 --no-build-isolation; RC[flash]=$?
[ "${RC[flash]}" -ne 0 ] && echo "[$(date +%T)] FLASH_ATTN_FAILED rc=${RC[flash]}"

echo "[$(date +%T)] STEP import smoke test"
"$PY" - <<'PY'
import importlib, sys
mods = ["torch", "torchaudio", "torchvision", "transformers", "decord",
        "liger_kernel", "deepspeed", "soundfile", "librosa", "huggingface_hub"]
for m in mods:
    try:
        x = importlib.import_module(m)
        print(f"OK  {m} {getattr(x,'__version__','?')}")
    except Exception as e:
        print(f"ERR {m}: {type(e).__name__}: {e}")
try:
    import flash_attn
    print("OK  flash_attn", flash_attn.__version__)
except Exception as e:
    print(f"WARN flash_attn unavailable: {type(e).__name__}: {e}")
try:
    import torch, torchaudio, torchvision
    print("TORCH_TRIO", torch.__version__, torchaudio.__version__, torchvision.__version__,
          "cuda_build", torch.version.cuda)
except Exception as e:
    print("ERR torch trio:", e); sys.exit(3)
PY
RC[smoke]=$?

echo "[$(date +%T)] RC SUMMARY:"
for k in torch codec stack flash smoke; do echo "  $k=${RC[$k]:-NA}"; done
if [ "${RC[torch]:-1}" -eq 0 ] && [ "${RC[stack]:-1}" -eq 0 ] && [ "${RC[smoke]:-1}" -eq 0 ]; then
  echo "[$(date +%T)] VS2 TORCH FIX DONE OK (flash=${RC[flash]:-NA})"
else
  echo "[$(date +%T)] VS2 TORCH FIX DONE WITH FAILURES"
fi
