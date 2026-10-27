"""Copy-vs-edit diagnostic for arbitrary sample folders (same statistics as
diag_copy_vs_edit.py, which has hard-coded roots).  Usage:
  python scripts/diag_copy_vs_edit_any.py NAME=path/to/ld_txt-3.0_ld_mot-3.0 [NAME2=...]
Prints per folder: N, median per-sample dynamics ratio (2nd-diff L1, gen/target),
mean d(gen,target), mean d(gen,aligned source), % of samples closer to source."""
import numpy as np, glob, sys

def stats(d):
    files = sorted(glob.glob(d + "/*.npy"))
    dyn, dt, ds, closer, n, dd = [], [], [], 0, 0, []
    for f in files:
        o = np.load(f, allow_pickle=True).item()
        g = np.asarray(o["pose"], float); t = np.asarray(o["pose_target"], float); sr = np.asarray(o["pose_source"], float)
        L = min(len(g), len(t))
        if L < 3: continue
        a = np.abs(g[2:L] - 2*g[1:L-1] + g[:L-2]).mean(); b = np.abs(t[2:L] - 2*t[1:L-1] + t[:L-2]).mean()
        if b > 1e-5: dyn.append(a/b)
        e_t = np.linalg.norm(g[:L] - t[:L], axis=-1).mean()
        idx = np.linspace(0, len(sr)-1, L).astype(int)
        e_s = np.linalg.norm(g[:L] - sr[idx], axis=-1).mean()
        idx_t = np.linspace(0, len(sr)-1, len(t)).astype(int)
        dd.append(np.linalg.norm(t - sr[idx_t], axis=-1).mean())
        dt.append(e_t); ds.append(e_s); closer += int(e_s < e_t); n += 1
    return n, dyn, dt, ds, closer, dd

print("%-28s %6s %10s %11s %11s %10s %12s" % ("method", "N", "dyn_ratio", "d(gen,tgt)", "d(gen,src)", "closerSRC", "d(tgt,src)"))
for arg in sys.argv[1:]:
    name, d = arg.split("=", 1)
    n, dyn, dt, ds, closer, dd = stats(d)
    if n == 0:
        print("%-28s  (no samples in %s)" % (name, d)); continue
    print("%-28s %6d %10.3f %11.4f %11.4f %9.1f%% %12.4f" % (
        name, n, float(np.median(dyn)), float(np.mean(dt)), float(np.mean(ds)), 100.0*closer/n, float(np.mean(dd))))
