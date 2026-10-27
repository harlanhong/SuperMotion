"""Edit single motions with a trained SuperMotion checkpoint.

    # use the dataset's own instruction
    python edit.py folder=checkpoints/supermotion_release keyid=000038

    # your own instruction, several source motions, stronger source preservation
    python edit.py folder=checkpoints/supermotion_release 'keyid=[000038,000091]' \
        text="raise both arms much higher" field_scale=1.5 out_dir=outputs

For every source motion this writes two files to `out_dir`:

  <keyid>.npy   {'pose', 'pose_source', 'pose_target', 'text'} in the 135-d feature format
                (3 translation + 6D root orientation + 21 joints x 6D), i.e. what `evaluate.py` saves
  <keyid>.npz   the same generated motion as SMPL-H parameters: `trans` (T,3),
                `global_orient` (T,3) and `body_pose` (T,63), axis-angle, 30 fps

The source motions come from the MotionFix dataset, so the usual data setup is required.
"""
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

import src.launch.prepare  # noqa: F401  (registers the Hydra/OmegaConf resolvers)
from evaluate import FEATURE_TYPES, build_sampler, load_model, prepare_test_batch

logger = logging.getLogger(__name__)


def to_smpl(pose):
    """135-d features -> SMPL-H axis-angle parameters."""
    from src.tools.transforms3d import transform_body_pose
    pose = torch.as_tensor(pose)
    T = pose.shape[0]
    return {'trans': pose[:, :3].numpy(),
            'global_orient': transform_body_pose(pose[:, 3:9], '6d->aa').reshape(T, 3).numpy(),
            'body_pose': transform_body_pose(pose[:, 9:].reshape(T, 21, 6), '6d->aa').reshape(T, 63).numpy()}


@torch.no_grad()
def run(cfg: DictConfig) -> None:
    import numpy as np
    import pytorch_lightning as pl
    from hydra.utils import instantiate

    from src.data.tools.collate import collate_batch_last_padding

    run_dir = Path(hydra.utils.to_absolute_path(cfg.folder))
    cfg = OmegaConf.merge(OmegaConf.load(run_dir / ".hydra/config.yaml"), cfg)
    out_dir = Path(hydra.utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    diffusion, steps = build_sampler(cfg, cfg.field_scale)
    logging.getLogger("pytorch_lightning.utilities.seed").setLevel(logging.WARNING)
    pl.seed_everything(cfg.seed)
    model = load_model(cfg.last_ckpt_path)

    data_module = instantiate(cfg.data, amt_only=True, load_splits=['test', 'val'])
    pool = {str(d['id']): d for split in ('test', 'val') for d in data_module.dataset[split].data}
    datasets = {str(d['id']): data_module.dataset[split]
                for split in ('test', 'val') for d in data_module.dataset[split].data}

    raw = cfg.keyid if isinstance(cfg.keyid, (list, ListConfig)) else str(cfg.keyid).split(',')
    keyids = [str(k).strip().zfill(6) for k in raw if str(k).strip()]
    unknown = [k for k in keyids if k not in pool]
    if unknown:
        raise SystemExit(f"unknown keyid(s): {unknown}. Ids come from the MotionFix test/val splits, "
                         f"e.g. {sorted(pool)[:3]}")

    items = []
    for k in keyids:
        ds = datasets[k]
        items.append(ds[[str(d['id']) for d in ds.data].index(k)])
    batch = collate_batch_last_padding(items, data_module.dataset['test'].load_feats)
    texts = [cfg.text if cfg.text else t for t in batch['text']]
    for k, t in zip(keyids, texts):
        logger.info(f"{k}: \"{t}\"")

    source_lens, target_lens = batch['length_source'], batch['length_target']
    input_batch = prepare_test_batch(model, batch)
    mask_source, mask_target = model.prepare_mot_masks(
        source_lens, target_lens, max_len=300 if model.pad_inputs else None)

    diffout = model.generate_motion([t.lower() for t in texts],
                                    input_batch['source_motion'].clone(),
                                    mask_source, mask_target, diffusion,
                                    source_lens, target_lens,
                                    init_vec=None, init_vec_method=cfg.init_from,
                                    condition_mode=cfg.condition_mode,
                                    gd_motion=cfg.guidance_scale_motion,
                                    gd_text=cfg.guidance_scale_text_n_motion,
                                    num_diff_steps=steps, inpaint_dict=None,
                                    use_linear=False, prob_way=cfg.prob_way)
    generated = model.diffout2motion(diffout)

    def gt_motion(which):
        n = len(keyids)
        single = [{k: (v[i:i + 1].to(model.device) if torch.is_tensor(v) else v[i:i + 1])
                   for k, v in batch.items() if which in k} for i in range(n)]
        return torch.cat([model.data2motion(s, FEATURE_TYPES, mot=which) for s in single], dim=0)

    source_motion, target_motion = gt_motion('source'), gt_motion('target')

    for i, k in enumerate(keyids):
        pose = generated[i, :target_lens[i]].cpu().numpy()
        np.save(out_dir / f"{k}.npy",
                {'pose': pose,
                 'pose_source': source_motion[i, :source_lens[i]].cpu().numpy(),
                 'pose_target': target_motion[i, :target_lens[i]].cpu().numpy(),
                 'text': texts[i]})
        np.savez(out_dir / f"{k}.npz", text=texts[i], fps=30, **to_smpl(pose))
    logger.info(f"wrote {len(keyids)} edited motions to {out_dir}")


@hydra.main(config_path="configs", config_name="edit", version_base="1.2")
def _main(cfg: DictConfig) -> None:
    return run(cfg)


if __name__ == '__main__':
    _main()
