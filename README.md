# Train HGERE model on AAAI Submission Dataset

## Installation and Preparation
 * We use `uv` to manage the installation

### Install depenendencies
 * install `uv` first
   * `curl -LsSf https://astral.sh/uv/install.sh | sh`
 * install `rust` to be able to install transformers (see below)
 * install bash if not available

### External Tools
 * We used wandb for logging. Please set up wandb if you want to use the logging functionality. This is not mandatory. (Use script parameter in train script to turn on or off)

### Tested Hardware
 * Linux Server with ubuntu
 * GPU Access A40 GPUs with 40Gig Ram
   * The HGERE is gpu memory hungry for large batch sizes. Batch size of 18 with 40Gig GPU works for us 

### install rust (Needed for transformers)
 * `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
 * `source "$HOME/.cargo/env"`

### Load scibert base model
 * install lfs for git 
   * `curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`
   * `sudo apt-get install git-lfs` 
 * outside base folder: `git clone https://huggingface.co/allenai/scibert_scivocab_uncased`
   * mv `scibert_scivocab_uncased` to `pretrained_models`

### Note on Cuda Version (12.8)
 * in `pyproject.toml` the installation of cuda 12.8 is used for other cuda version you need to set up things on your own.

### install dependencies via uv
 * `uv sync`
 * activate uv virtual environment `source .venv/bin/activate
 * You are ready to use the bash scripts in `scripts` folder

## How HGERE works
HGERE contains two steps based on two different models. First is a pruner model, which is trained on the NER annotations but is not considering the NER labels. It reduces the candidates for NER from 12 * n (where n is the number of words in the sentences) to ~5 candidates in mean.

The first step is to train the pruner. The pruner results are then used as input for the HGERE Entity and Relation Extraction (ERE) model

### Run pruner and hgere training
 * Use `GPU_ID` parameter in scripts to choose specific GPU(s) on your machine.
 * Use the scripts:
   * `scripts/pruner/train.py`
   * `scripts/hgere/train.py`


