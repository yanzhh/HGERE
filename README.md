# HGERE

Source Code for "Joint Entity and Relation Extraction with Span Pruning and Hypergraph Neural Networks" of EMNLP2023

Our code is based on [PL-marker](https://github.com/thunlp/PL-Marker), thanks for their great work!

## Setup

### Install Dependencies/Prepare the datasets

Please refer to [PL-marker](https://github.com/thunlp/PL-Marker) for dependency installation and dataset preparing instructions.

### Download Pre-trained Language Models

Download scibert_scivocab_uncased, bert-base-uncased and albert-xxlarge-v1 from huggingface, and place them in fold "pretrained_models".

## Training

### Pruners

You can use run_pruner.py to train an entity pruner model, we train with five seeds and use the pruner model with highest dev recall for downstream process.

Please run scripts in the fold: shells/pruner, for example:

```shellsession
bash shells/pruner/scierc/run_train_pruner_scierc.sh
...
```

### ERE models

Then we can train the ERE models with ent_pred_{train/dev/test}.json files in the output directory of the span pruner. 

For example we can train the HGERE models of scierc dataset with command as follow:

```shell
bash shells/hgere/scierc/run_train_scierc_scibert_hgere.sh
```

 All the scripts are in the fold: shells/hgere

## Evaluate the pre-trained ERE models

You can replace "--do_train --do_eval" in the scripts for training with "--do_test" if you need to evaluate on a pre-trained ERE model.

## Citation

If you use our code in your research, please cite our work:

```shell
@misc{yan2023joint,
      title={Joint Entity and Relation Extraction with Span Pruning and Hypergraph Neural Networks}, 
      author={Zhaohui Yan and Songlin Yang and Wei Liu and Kewei Tu},
      year={2023},
      eprint={2310.17238},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
