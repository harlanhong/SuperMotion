#!/usr/bin/env bash
# Downloads everything inference needs: the MotionFix dataset, the TMR evaluator, the SMPL-H body
# models and the CLIP text encoder. Run from the repository root.
set -e
pip install -q gdown
mkdir -p data/body_models deps
# MotionFix dataset (motions, splits, test/val libraries)
gdown --folder "https://drive.google.com/drive/folders/1DM7oIJwxwoljVxAfhfktocTptwVX5sqR?usp=sharing"
mv motionfix-dataset data/
# TMR retrieval evaluator (weights + stats)
gdown --folder "https://drive.google.com/drive/folders/15LHeriOCjmh4Cp5H9M94xoFGBN0amxI8?usp=sharing"
mv tmr-evaluator eval-deps
# SMPL-H body models (neutral)
gdown --folder "https://drive.google.com/drive/folders/1s3re2I1OzBimQIpudUEFB1hClFWOPJjC?usp=drive_link"
mv smplh data/body_models/
# similarity curves + feature statistics, from the SimMotionEdit release folder
gdown --folder "https://drive.google.com/drive/folders/1LjiKVjDHqOEnykZMP3ZiTJEz_EY-TqC3?usp=sharing" -O simmotionedit-release || true
[ -f simmotionedit-release/scores_v3.npz ] && mv simmotionedit-release/scores_v3.npz data/
# CLIP text encoder
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("openai/clip-vit-large-patch14", local_dir="deps/clip-vit-large-patch14",
                  local_dir_use_symlinks=False)
PY
echo "expected layout: data/motionfix-dataset, data/body_models/smplh, data/scores_v3.npz,"
echo "                 deps/clip-vit-large-patch14, deps/stats/statistics_motionfix.npy, eval-deps/"
