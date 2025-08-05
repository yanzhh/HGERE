# Reproducability changes from HGERE original version
Changes to HGERE version to be able to run

## Use uv for installation management
 * first install uv
 * `uv init`

## install rust (Needed for transformers)
 * `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
 * `source "$HOME/.cargo/env"`

## other dependencies:
* uv add torch
* uv add wandb
* uv add setuptools

## dependencies not used any more
 * apex
 * tensorboardx

## install transformers from this repo
* uv add --editable ./transformers

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


## Next steps to test the repo:
 * first train pruner
 * then use pruner to predict candidates (--do-test --output-results)
 * then use pruner results for training HGERE

