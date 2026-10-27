#!/usr/bin/env bash
# Creates ./mfix-env with the dependencies the released checkpoint was evaluated with.
set -e
python3.10 -m venv mfix-env
source "$PWD/mfix-env/bin/activate"
pip install --upgrade pip setuptools
pip install "torch==2.1.2" "torchvision==0.16.2" --index-url https://download.pytorch.org/whl/cu118
pip install "pytorch-lightning==2.2.4"
pip install -r requirements.txt
echo "done. activate with: source mfix-env/bin/activate"
