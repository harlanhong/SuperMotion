#!/usr/bin/env bash
# End-to-end check of an installation: edits one motion, samples a few, scores them.
# usage: bash scripts/selftest.sh [CKPT_DIR=checkpoints/supermotion_release]
set -e
CKPT=${1:-checkpoints/supermotion_release}
export WANDB_MODE=${WANDB_MODE:-disabled} HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== 1/3  data, checkpoint and dependencies"
python - "$CKPT" <<'PY'
import sys, os
ckpt = sys.argv[1]
need = [f"{ckpt}/.hydra/config.yaml", f"{ckpt}/checkpoints/last.ckpt",
        "data/motionfix-dataset/motionfix.pth.tar", "data/body_models/smplh",
        "data/scores_v3.npz", "deps/clip-vit-large-patch14", "eval-deps",
        "assets/motionfix_test_ids.json"]
missing = [p for p in need if not os.path.exists(p)]
print("missing:", missing if missing else "nothing")
sys.exit(1 if missing else 0)
PY

echo "== 2/3  edit one motion"
python edit.py folder="$CKPT" keyid=000038 out_dir="$TMP/edit" >"$TMP/edit.log" 2>&1 || { tail -20 "$TMP/edit.log"; exit 1; }
python - "$TMP/edit" <<'PY'
import numpy as np, sys, os
d = np.load(os.path.join(sys.argv[1], "000038.npy"), allow_pickle=True).item()
z = np.load(os.path.join(sys.argv[1], "000038.npz"), allow_pickle=True)
assert np.isfinite(d['pose']).all() and d['pose'].shape[1] == 135, "bad features"
assert z['body_pose'].shape[1] == 63 and len(z['trans']) == len(d['pose']), "bad SMPL export"
print(f"  generated {d['pose'].shape[0]} frames | instruction: \"{d['text']}\"")
PY

echo "== 3/3  sample 32 motions and score them"
python evaluate.py folder="$CKPT" max_samples=32 batch_size=32 out_tag=_selftest >"$TMP/eval.log" 2>&1 || { tail -20 "$TMP/eval.log"; exit 1; }
S="$PWD/$CKPT/3way_steps_300_motionfix_noise_last_selftest/ld_txt-3.0_ld_mot-3.0"
echo "  $(ls $S/*.npy | wc -l) motions written"
python scripts/dyn_ratio.py selftest="$S"
rm -rf "$PWD/$CKPT/3way_steps_300_motionfix_noise_last_selftest"
echo "OK - the installation works. Reproduce the paper table with: python evaluate.py folder=$CKPT"
