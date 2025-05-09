
* working until python3.9

* install uv
* uv init
## install rust
 * `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
 * `source "$HOME/.cargo/env"`

## other dependencies:
* uv add torch
* uv add tensorboardX
* uv add setuptools

## install transformers
* uv add --editable ./transformers
## install apex
 * installed apex from a fresh clone:
```
git clone https://github.com/NVIDIA/apex
cd apex
uv pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings '"--build-option=--cpp_ext"' --config-settings '"--build-option=--cuda_ext"' ./
```

## load models from huggingface into pretrained_models
 * install lfs for git 
   * `curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`
   * `sudo apt-get install git-lfs` 
 * outside base folder: `git clone https://huggingface.co/allenai/scibert_scivocab_uncased`
   * mv scibert_scivocab_uncased to pretrained_models


## run scierc pruner
### delete import AlberForSpanPruner (does not exist) in run_pruner.py

### for pruner only one GPU is working
 * not tested: distributed over different server
 * But assigning more than one gpu is not working.
### Changes in bash scipt
 * change bashscript (incl. correct data folder)
 * try to run without gpu
### Run bashscript unsing CUDA_VISIBLE_DEVICES and nice
   * `CUDA_VISIBLE_DEVICES= nice -n 10 ./shells/pruner/scierc/run_train_pruner_scierc.sh`


## Process:
 * first train pruner
 * then use pruner to predict candidates (--do-test --output-results) (@todo is this working? )
 * then use pruner results for training HGERE

