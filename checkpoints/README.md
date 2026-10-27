# Checkpoint

```bash
cd checkpoints
wget https://github.com/harlanhong/SuperMotion/releases/download/v1.0/supermotion_release.tar.gz
tar xzf supermotion_release.tar.gz
```

gives

```
checkpoints/supermotion_release/
  .hydra/config.yaml        architecture and training configuration (read by the scripts)
  .hydra/overrides.yaml
  checkpoints/last.ckpt     186 MB
```

Then `python edit.py folder=checkpoints/supermotion_release keyid=000038`.

The file is 186 MB rather than 2.2 GB because the frozen CLIP text encoder is not duplicated into it.
It is rebuilt from `deps/clip-vit-large-patch14`, so run `scripts/download_data.sh` first; the assembled
model is bit-identical to the one the paper was evaluated with. `.ckpt` files are git-ignored.
