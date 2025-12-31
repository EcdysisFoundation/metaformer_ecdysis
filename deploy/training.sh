#!/bin/bash

# This script is used to train a new model.

set -eE  # Exit if any command fails https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/


cd /home/ecdysis/MetaFormer/

MODEL_PREFIX="output/ecdysis/$1"
THIS_VERSION="$2"

export GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "This new version is: ${THIS_VERSION}"
echo "Found ${GPU_COUNT} GPU(s)"

DATASET="bugbox_model_${THIS_VERSION}"

# Copy dataset report from dataset to model folder
. ./deploy/copy_reports.sh "${DATASET}" "${MODEL_PREFIX}" "${THIS_VERSION}"

# Run training starting from last best checkpoint
/home/ecdysis/miniconda3/envs/pytorch/bin/torchrun --nproc_per_node ${GPU_COUNT} main.py --cfg configs/ecdysis.yaml \
 --data-path "datasets/${DATASET}/" --tag "$1" --version "$THIS_VERSION" \
  --pretrain "output/ecdysis/morphospecies/${DEPLOYED_VERSION}/best.pth" >/dev/null
wait
# Evaluate trained model
/home/ecdysis/miniconda3/envs/pytorch/bin/torchrun --nproc_per_node ${GPU_COUNT} main.py \
  --cfg "${MODEL_PREFIX}/${THIS_VERSION}/config.yaml" --dataset bugbox --data-path "datasets/${DATASET}" --eval  --pretrain "${MODEL_PREFIX}/${THIS_VERSION}/best.pth" --version "$THIS_VERSION" > "${MODEL_PREFIX}/${THIS_VERSION}/console_output.txt"
wait
