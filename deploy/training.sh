#!/bin/bash

# This script is used to train a new model.

set -eE  # Exit if any command fails https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/

# run script, writing output to log
# bash deploy/training.sh DATASET_NAME STARTING_CHECKPOINT_DIR THIS_VERSION > training_run.log 2>&1 &
# example, note 6/6 is refering to the directory structure where dataset=6, previous version = 6
#          This assumes a directory based on the dataset, with a subdirectory based on the version
# bash deploy/training.sh 6 6/6 testing6 > training_run.log 2>&1 &
# view training log for status updates
# tail -f training_run.log

cd /home/ecdysis/MetaFormer/

OUTPUT_DIR="output/ecdysis"
DATASET_NAME="$1"
STARTING_CHECKPOINT_DIR="$2"
THIS_VERSION="$3"

export PYTHONUNBUFFERED=1
export GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "This new version is: ${THIS_VERSION}"
echo "Found ${GPU_COUNT} GPU(s)"

# Copy dataset report from dataset to model folder
. ./deploy/copy_reports.sh "${DATASET_NAME}" "${OUTPUT_DIR}" "${THIS_VERSION}"

# Run training starting from last best checkpoint
/home/ecdysis/miniconda3/envs/pytorch/bin/torchrun --nproc_per_node ${GPU_COUNT} main.py --cfg configs/ecdysis.yaml \
 --data-path "datasets/${DATASET_NAME}/" --tag "$1" --version "$THIS_VERSION" \
  --pretrain "${OUTPUT_DIR}/${STARTING_CHECKPOINT_DIR}/best.pth"
wait
# Evaluate trained model
/home/ecdysis/miniconda3/envs/pytorch/bin/torchrun --nproc_per_node ${GPU_COUNT} main.py \
  --cfg "${OUTPUT_DIR}/${DATASET_NAME}/${THIS_VERSION}/config.yaml" --dataset bugbox --data-path "datasets/${DATASET_NAME}" --eval  --pretrain "${OUTPUT_DIR}/${DATASET_NAME}/${THIS_VERSION}/best.pth" --version "$THIS_VERSION"
wait

/home/ecdysis/miniconda3/envs/pytorch/bin/torchrun --nproc_per_node 2 main.py \
  --cfg "output/ecdysis/6/7/config.yaml" --dataset bugbox --data-path "datasets/6" --eval  --pretrain "output/ecdysis/6/7/best.pth" --version "7"
