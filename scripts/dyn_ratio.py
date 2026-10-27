"""Aggregate dynamics ratios gen/target over a sample folder (the DYNAMICS line of the evaluation recipe):
  root = mean root-translation speed, vel = mean |d pose|, hf = mean |d^2 pose| (pose = 6d rots, cols 9:).
Usage: python scripts/dyn_ratio.py NAME=folder [NAME2=folder2 ...]"""
import numpy as np, glob, sys
for arg in sys.argv[1:]:
    name, d = arg.split("=", 1)
    fs = sorted(glob.glob(f"{d}/*.npy")); grv=gjv=ghv=trv=tjv=thv=0.
    for f in fs:
        x = np.load(f, allow_pickle=True).item(); g, t = x["pose"], x["pose_target"]
        grv += np.linalg.norm(np.diff(g[:,:3],axis=0),axis=1).mean(); trv += np.linalg.norm(np.diff(t[:,:3],axis=0),axis=1).mean()
        gjv += np.abs(np.diff(g[:,9:],axis=0)).mean(); tjv += np.abs(np.diff(t[:,9:],axis=0)).mean()
        ghv += np.abs(np.diff(g[:,9:],n=2,axis=0)).mean(); thv += np.abs(np.diff(t[:,9:],n=2,axis=0)).mean()
    if not fs: print(f"DYNAMICS {name} N=0 (no samples in {d})"); continue
    print(f"DYNAMICS {name} N={len(fs)} root={grv/trv:.3f} vel={gjv/tjv:.3f} hf={ghv/thv:.3f}")
