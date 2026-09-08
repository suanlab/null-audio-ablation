#!/bin/bash
# Script to finalize experiments: extract MVBench result, update docs, and migrate checkpoints
# Run after No-Bridge MVBench eval completes

set -e
cd $REPO_ROOT

RESULT_FILE="eval_results/mvbench_no_bridge.json"
NAS_DEST="/mnt/nas/Users/suan/VideoLLM/checkpoints"

echo "=== Step 1: Check MVBench Result ==="
if [ ! -f "$RESULT_FILE" ]; then
    echo "ERROR: $RESULT_FILE not found. Eval may still be running."
    exit 1
fi

echo "Result file found. Extracting accuracy..."
OVERALL=$(python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
print(f\"{d.get('overall_accuracy', d.get('accuracy', 'N/A'))}\")
")
echo "No-Bridge MVBench: $OVERALL"

echo ""
echo "=== Step 2: Migrate Active Checkpoints to NAS ==="
echo "Source checkpoints (local):"
du -sh checkpoints/full_v6 checkpoints/ablation checkpoints/audio_quick_bridge_v2 2>/dev/null

echo ""
echo "Destination: $NAS_DEST"
echo "NAS free space:"
df -h /mnt/nas/ | tail -1

echo ""
echo "Migrating full_v6..."
rsync -avh --progress checkpoints/full_v6/ "$NAS_DEST/full_v6/"

echo ""
echo "Migrating ablation (vision_only + no_bridge)..."
rsync -avh --progress checkpoints/ablation/ "$NAS_DEST/ablation/"

echo ""
echo "Migrating audio_quick_bridge_v2..."
rsync -avh --progress checkpoints/audio_quick_bridge_v2/ "$NAS_DEST/audio_quick_bridge_v2/"

echo ""
echo "=== Step 3: Verify Migration ==="
echo "Local checkpoints remaining:"
du -sh checkpoints/ 2>/dev/null || echo "checkpoints/ not found (already removed)"
echo "NAS checkpoints:"
du -sh "$NAS_DEST/" 2>/dev/null

echo ""
echo "=== Done ==="
echo "No-Bridge MVBench: $OVERALL"
echo "All active checkpoints migrated to NAS."
echo ""
echo "NOTE: Remove local checkpoints manually after verifying NAS copy:"
echo "  rm -rf checkpoints/full_v6 checkpoints/ablation checkpoints/audio_quick_bridge_v2"
