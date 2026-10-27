# SuperMotion — inference code

Text-driven 3D human motion editing: given a **source motion** and an **edit instruction**, produce the
edited motion while preserving everything the instruction does not ask to change.

This repository contains the **inference code and the released checkpoint** for
[SuperMotion: Source-Preserving Denoising for Text-Driven Human Motion Editing](https://arxiv.org/abs/2610.01517)
(arXiv:2610.01517). Training code is not included.

The model is a v-prediction DiT with U-ViT long skip connections and a **preservation field**
g ∈ (0,1)^(T×D). At every reverse step the clean-motion estimate is re-anchored to the time-aligned
source,

```
x̂₀* = x̂₀ + g ⊙ (align(source) − x̂₀)
```

in clean-motion space (never in v-space, where the correction would be divided by √(1−ᾱₜ)). The field is
learned from a single source-anchor loss without localisation labels: it decides *which degrees of
freedom* may be copied from the source, and copies more when the edit is small.

## Results — MotionFix

Retrieval with the TMR evaluator, generated → target. **B** = within batches of 32, **F** = over the full
test pool. AvgR is better when lower.

| | B R@1 | B R@2 | B R@3 | B AvgR | F R@1 | F R@2 | F R@3 | F AvgR |
|---|---|---|---|---|---|---|---|---|
| SimMotionEdit (CVPR'25) | 70.62 | 82.92 | 88.12 | 2.38 | 25.49 | 39.33 | 49.21 | 23.49 |
| OmniME (CVPR'26) | 77.29 | 88.54 | 91.88 | 1.79 | 32.02 | 50.20 | 59.88 | 13.06 |
| **SuperMotion** (paper) | **79.17** | **89.17** | **92.92** | **1.68** | **33.20** | **50.79** | 59.09 | **12.87** |
| SuperMotion — reproduced with this code | 78.96 | 89.58 | 92.50 | 1.71 | 33.60 | 49.41 | 59.29 | 13.16 |

The last row is the output of `python evaluate.py folder=checkpoints/supermotion_release` in this
repository, verified on one A100. It differs from the paper row only through the noise draw (see
[Reproducibility](#4-reproducibility)); repeated runs of this code with the same settings are bit-identical.
The same run gives dynamics ratios of 0.84 (root) / 0.92 (velocity) / 0.90 (high frequency) against the
ground truth and a per-sequence detail ratio of 1.02, where 1.0 is the ground truth.

## 1. Setup

Tested with Python 3.10, PyTorch 2.1.2 + CUDA 11.8 and one A100 (any GPU with ≥ 8 GB works for
inference).

```bash
git clone https://github.com/harlanhong/SuperMotion.git && cd SuperMotion
bash scripts/install.sh          # creates ./mfix-env with the pinned dependencies
source mfix-env/bin/activate
```

`requirements-lock.txt` is the exact environment every number above was produced with. If you build the
environment yourself, keep `transformers==4.17` together with `huggingface-hub==0.17.3`; newer hub
releases are not compatible with that version of transformers.

### Data and dependencies

```bash
bash scripts/download_data.sh    # MotionFix dataset, TMR evaluator, SMPL-H bodies, CLIP text encoder
```

Expected layout:

```
data/motionfix-dataset/     motionfix.pth.tar, motionfix_test.pth.tar, splits.json, ...
data/body_models/smplh/     SMPL-H neutral body model
data/scores_v3.npz          per-frame source-target similarity curves (from the SimMotionEdit release)
deps/clip-vit-large-patch14/  CLIP text encoder (openai/clip-vit-large-patch14)
deps/stats/statistics_motionfix.npy   feature mean/std
eval-deps/                  TMR retrieval evaluator (weights + stats)
```

`data/scores_v3.npz` is only read when the dataset is built; the model itself does not use it at
inference. `deps/stats/statistics_motionfix.npy` is recomputed from the data if missing.

### Checkpoint

Download the released checkpoint into `checkpoints/`:

```
checkpoints/supermotion_release/
  .hydra/config.yaml        architecture and training configuration (read by the scripts)
  .hydra/overrides.yaml
  checkpoints/last.ckpt     2.2 GB
```

## 2. Edit a motion

```bash
# apply the instruction that ships with the motion
python edit.py folder=checkpoints/supermotion_release keyid=000038

# your own instruction, several source motions at once
python edit.py folder=checkpoints/supermotion_release 'keyid=[000038,000091]' \
    text="raise both arms much higher" out_dir=outputs
```

Source motions are taken from the MotionFix test/val splits by id. For each one you get

| file | contents |
|---|---|
| `<keyid>.npy` | `{'pose', 'pose_source', 'pose_target', 'text'}`, each motion `(T, 135)`: 3 translation + 6D root orientation + 21 joints × 6D |
| `<keyid>.npz` | the generated motion as SMPL-H parameters: `trans (T,3)`, `global_orient (T,3)`, `body_pose (T,63)`, axis-angle, 30 fps |

The 6D rotations the network outputs are not exactly orthonormal; they are projected onto SO(3) when
exported to SMPL-H, so the `.npz` is what you should render.

## 3. Reproduce the benchmark

```bash
python evaluate.py folder=checkpoints/supermotion_release          # ~25 min on one A100
S=$PWD/checkpoints/supermotion_release/3way_steps_300_motionfix_noise_last/ld_txt-3.0_ld_mot-3.0
python compute_metrics.py folder=$S                                # retrieval metrics
python scripts/dyn_ratio.py release=$S                             # dynamics ratios against the ground truth
python scripts/diag_copy_vs_edit_any.py release=$S                 # copy-vs-edit diagnostic
```

`evaluate.py` samples the 836 evaluation motions (506 test + 330 val) and writes one `.npy` per motion,
in the same format as `edit.py`. `compute_metrics.py` scores the test ids among them.

Useful overrides:

| override | meaning |
|---|---|
| `guidance_scale_text_n_motion`, `guidance_scale_motion` | classifier-free guidance weights (default 3.0 / 3.0) |
| `field_scale` | strength of the preservation field at inference; 1.0 = as trained |
| `batch_size`, `max_samples` | batching; stop after N motions (quick check) |
| `out_tag` | suffix for the output folder, to keep several settings side by side |

`field_scale` is a dial between editing and copying. Raising it keeps more of the source: top-1
retrieval against the target keeps rising (34.6 at ×4), but so does retrieval against the *source*
(46.6 at ×4 versus 38.5 at ×1), i.e. the gain is copying rather than editing. 1.0 is the operating
point reported in the paper.

## 4. Reproducibility

MotionFix evaluates on **half of the test split**. Upstream draws that half with the global Python RNG
at the moment the dataset is built, which makes the evaluated pool depend on everything that happened to
consume randomness earlier in the process — even the number of log lines printed. This repository
instead reads the pool from `assets/motionfix_test_ids.json` (the 506 motions the paper reports on) and
warns if the pool it ends up with differs. Set `data.preproc.test_ids_file=null` to restore the
upstream behaviour.

Generation itself is stochastic. With a fixed pool, fixed `batch_size` and the seed stored in the
checkpoint config, repeated runs produce identical motions; changing the batch size changes which noise
sample each motion gets, and therefore shifts the metrics by a few tenths of a point.

## 5. Repository layout

```
edit.py                     edit single motions
evaluate.py                 sample the evaluation set
compute_metrics.py          TMR retrieval metrics
configs/                    Hydra configs (edit.yaml, motionfix_eval.yaml, data/)
assets/                     the fixed evaluation pool
src/model/                  base_diffusion.py (sampling), DiT_denoiser_cls_arch.py (denoiser + field head), DiT_models.py (DiT with long skips)
src/diffusion/              gaussian_diffusion.py (v-prediction, the clean-space anchor inside the sampler)
src/data/                   MotionFix data module and motion features
tmr_evaluator/, eval-deps/  retrieval evaluator
scripts/                    install.sh, download_data.sh, selftest.sh and the two diagnostics
```

Check an installation end to end with `bash scripts/selftest.sh` (a few minutes: it verifies the data
layout, edits one motion and samples 32).

## License

MIT (`LICENSE`); the MotionFix components keep their own license (`LICENSE-MF`). The SMPL-H body models,
the MotionFix dataset, CLIP and the TMR evaluator are covered by their respective licenses.

## Citation

```bibtex
@article{hong2026supermotion,
  title   = {SuperMotion: Source-Preserving Denoising for Text-Driven Human Motion Editing},
  author  = {Fa-Ting Hong and Peter Wonka},
  journal = {arXiv preprint arXiv:2610.01517},
  year    = {2026}
}
```

## Acknowledgements

Built on [SimMotionEdit](https://github.com/lzhyu/SimMotionEdit) (Li et al., CVPR 2025) and
[MotionFix](https://github.com/atnikos/motionfix) (Athanasiou et al., SIGGRAPH Asia 2024), whose data
pipeline and TMR evaluator this code reuses.
