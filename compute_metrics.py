"""TMR retrieval metrics for a folder of generated motions.

    python compute_metrics.py folder=/abs/path/to/ld_txt-3.0_ld_mot-3.0

Reports R@1/2/3 and the average rank, generated -> target (the edit metric) and source -> target,
within batches of 32 and over the full test pool. `folder` must be absolute.
"""
import glob
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm


def collect_gen_samples(motion_gen_path, normalizer, device):
    """Load generated motions and map them into the feature layout the evaluator expects."""
    from src.data.features import _get_body_transl_delta_pelv_infer

    samples, samples_raw = {}, {}
    print("Collecting generated samples")
    for fname in tqdm(glob.glob(f'{motion_gen_path}/*.npy')):
        keyid = Path(fname).name.replace('.npy', '')
        motion = torch.from_numpy(np.load(fname, allow_pickle=True).item()['pose'])
        trans, global_orient_6d, body_pose_6d = motion[..., :3], motion[..., 3:9], motion[..., 9:]
        trans_delta = _get_body_transl_delta_pelv_infer(global_orient_6d, trans)
        # evaluator layout: translation delta | joint rotations | root orientation
        samples[keyid] = normalizer(torch.cat([trans_delta, body_pose_6d, global_orient_6d], dim=-1)).to(device)
        samples_raw[keyid] = torch.cat([trans, global_orient_6d, body_pose_6d], dim=-1).to(device)
    return samples, samples_raw


def compute_metrics(cfg: DictConfig) -> None:
    from tmr_evaluator.motion2motion_retr import retrieval
    batches, full = retrieval(cfg.folder)
    print("\n===== Retrieval within batches of 32 =====")
    print(batches)
    print("\n===== Retrieval over the full test set =====")
    print(full)


@hydra.main(config_path="configs", version_base="1.2", config_name="compute_metrics")
def _main(cfg: DictConfig):
    return compute_metrics(cfg)


if __name__ == '__main__':
    _main()
