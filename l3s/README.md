# Train the models on l3s server
 * we need multiple seeds!

## What we need on the l3s server:
 * resources via scp
  * the pruner models for the pruned train data and inference
  * the rules.json for inference pre-filtering
 * config files via git

## train using slurm in project folder

`sbatch slurm/train-scier-l3s.sh

## How to move the results back:
 * move the datasets folders (they contain the predictions) back to gesis




## Example to move prediction back:


SSH_USER=l3shannover
PROJECT_PATH=/home/???/wolf??
PATH_PRED_SOURCE=$SSH_USER:PROJECT_PATH/datasets/
PATH_PRED_TARGET=/data_ssds/disk02/ottowg/??


scp -r PATH_PRED_SOURCE PATH_PRED_TARGET



