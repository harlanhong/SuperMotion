"""Sample edited motions for the MotionFix test set with a trained SuperMotion checkpoint.

    python evaluate.py folder=checkpoints/supermotion_release \
        guidance_scale_text_n_motion=3.0 guidance_scale_motion=3.0

`folder` is a run directory holding `.hydra/config.yaml` and `checkpoints/last.ckpt`.
Generated motions are written to

    <folder>/3way_steps_<steps>_<dataset>_noise_last/ld_txt-<g>_ld_mot-<g>/<keyid>.npy

as dicts with the generated motion, the source, the target and the edit instruction
(135-dimensional SMPL-H features per frame: translation, 6D root orientation, 6D joint rotations).
Score them with `python compute_metrics.py folder=<that directory>`.

Useful overrides: `field_scale` (strength of the preservation field at inference, 1.0 = as trained),
`max_samples` (stop after N motions, for a quick check), `out_tag` (suffix for the output folder).
"""
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import src.launch.prepare  # noqa: F401  (registers the Hydra/OmegaConf resolvers)

logger = logging.getLogger(__name__)

# the five feature groups that make up the 207-dimensional model input
FEATURE_TYPES = ['body_transl_delta_pelv', 'body_orient_xy', 'z_orient_delta',
                 'body_pose', 'body_joints_local_wo_z_rot']


def prepare_test_batch(model, batch):
    batch = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
    for k, v in model.norm_and_cat(batch, model.input_feats).items():
        batch[f'{k}_motion'] = v
    return batch


def build_sampler(cfg: DictConfig, field_scale: float):
    """The reverse process the checkpoint was trained with."""
    from src.diffusion import create_diffusion
    steps = cfg.model.diff_params.num_train_timesteps
    if cfg.num_sampling_steps is not None and cfg.num_sampling_steps < steps:
        steps = cfg.num_sampling_steps
    diffusion = create_diffusion(timestep_respacing=None,
                                 learn_sigma=False,
                                 sigma_small=True,
                                 diffusion_steps=steps,
                                 noise_schedule=cfg.model.diff_params.noise_schedule,
                                 predict_type=cfg.model.diff_params.predict_type)
    diffusion.field_scale = float(field_scale)
    return diffusion, steps


def load_model(ckpt_path: str):
    from src.model.base_diffusion import MD
    model = MD.load_from_checkpoint(ckpt_path, renderer=None, strict=False)
    model.eval()
    model.freeze()
    transl_feats = [x for x in model.input_feats if 'transl' in x]
    if set(transl_feats).issubset({"body_transl_delta", "body_transl_delta_pelv",
                                   "body_transl_delta_pelv_xy"}):
        model.using_deltas_transl = True
    logger.info(f"loaded {ckpt_path}")
    return model


def check_test_pool(data_module, cfg) -> None:
    """Warn if the evaluated test pool is not the one the paper reports (see the note in `sample`)."""
    ids_file = cfg.data.preproc.get('test_ids_file', None)
    ids_file = Path(ids_file) if ids_file else None
    if ids_file is None or not ids_file.exists():
        return
    import json
    expected = set(json.load(open(ids_file)))
    got = {str(d['id']) for d in data_module.dataset['test'].data}
    if got == expected:
        logger.info(f"test pool matches {ids_file.name} ({len(got)} motions)")
    else:
        logger.warning(f"test pool differs from {ids_file.name}: {len(got & expected)}/{len(expected)} shared ids. "
                       "Retrieval numbers will not match the paper exactly.")


@torch.no_grad()
def sample(cfg: DictConfig) -> None:
    import numpy as np
    import pytorch_lightning as pl
    from hydra.utils import instantiate
    from tqdm import tqdm

    from src.data.tools.collate import collate_batch_last_padding

    run_dir = Path(hydra.utils.to_absolute_path(cfg.folder))
    # the run's own config carries the architecture; the values passed here override it
    cfg = OmegaConf.merge(OmegaConf.load(run_dir / ".hydra/config.yaml"), cfg)

    # NOTE: the order of the next three blocks is load-bearing. MotionFix evaluates on a random
    # half of the test split drawn from the *global* RNG when the data module is built (upstream
    # behaviour, kept so that the released numbers are reproducible). Seeding, then building the
    # model, then the data module is what fixes the 506-motion pool; `check_test_pool` below
    # verifies the result against the id list the paper used.
    diffusion, steps = build_sampler(cfg, cfg.field_scale)

    out_dir = run_dir / (f'{cfg.prob_way}_steps_{steps}_{cfg.data.dataname}'
                         f'_{cfg.init_from}_{cfg.ckpt_name}{cfg.out_tag}')
    guid_text, guid_motion = cfg.guidance_scale_text_n_motion, cfg.guidance_scale_motion
    out_dir = out_dir / f'ld_txt-{guid_text}_ld_mot-{guid_motion}'
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"writing samples to {out_dir}")

    logging.getLogger("pytorch_lightning.utilities.seed").setLevel(logging.WARNING)
    pl.seed_everything(cfg.seed)

    model = load_model(cfg.last_ckpt_path)
    logger.info(f"{steps} sampling steps | guidance {guid_text}/{guid_motion} | field scale {diffusion.field_scale}")

    data_module = instantiate(cfg.data, amt_only=True, load_splits=['test', 'val'])
    check_test_pool(data_module, cfg)
    dataset = data_module.dataset['test'] + data_module.dataset['val']
    feats = data_module.dataset['test'].load_feats
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=8, batch_size=cfg.batch_size,
                                         collate_fn=lambda b: collate_batch_last_padding(b, feats))
    logger.info(f"evaluation set: {len(dataset)} motions")

    done = 0
    for batch in tqdm(loader):
        source_lens, target_lens = batch['length_source'], batch['length_target']
        n = len(batch['id'])

        # ground-truth source and target in the 135-d SMPL-H feature space, for the saved files
        def to_motion(which):
            single = [{k: (v[i:i + 1].to(model.device) if torch.is_tensor(v) else v[i:i + 1])
                       for k, v in batch.items() if which in k} for i in range(n)]
            return torch.cat([model.data2motion(s, FEATURE_TYPES, mot=which) for s in single], dim=0)

        source_motion, target_motion = to_motion('source'), to_motion('target')

        input_batch = prepare_test_batch(model, batch)
        mask_source, mask_target = model.prepare_mot_masks(
            source_lens, target_lens, max_len=300 if model.pad_inputs else None)

        diffout = model.generate_motion([t.lower() for t in batch['text']],
                                        input_batch['source_motion'].clone(),
                                        mask_source, mask_target, diffusion,
                                        source_lens, target_lens,
                                        init_vec=None, init_vec_method=cfg.init_from,
                                        condition_mode=cfg.condition_mode,
                                        gd_motion=guid_motion, gd_text=guid_text,
                                        num_diff_steps=steps, inpaint_dict=None,
                                        use_linear=False, prob_way=cfg.prob_way)
        generated = model.diffout2motion(diffout)

        for i in range(generated.shape[0]):
            np.save(out_dir / f"{str(batch['id'][i]).zfill(6)}.npy",
                    {'pose': generated[i, :target_lens[i]].cpu().numpy(),
                     'pose_source': source_motion[i, :source_lens[i]].cpu().numpy(),
                     'pose_target': target_motion[i, :target_lens[i]].cpu().numpy(),
                     'text': batch['text'][i]})
        done += n
        if cfg.max_samples is not None and done >= cfg.max_samples:
            logger.info(f"stopping after {done} motions (max_samples={cfg.max_samples})")
            break

    logger.info(f"done: {done} motions in {out_dir}")


@hydra.main(config_path="configs", config_name="motionfix_eval", version_base="1.2")
def _main(cfg: DictConfig) -> None:
    return sample(cfg)


if __name__ == '__main__':
    _main()
