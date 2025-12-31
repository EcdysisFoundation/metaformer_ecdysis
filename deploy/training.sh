#!/bin/bash

# This script is used to train a new model.

set -eE  # Exit if any command fails https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/


cd /home/ecdysis/MetaFormer/

OUTPUT_DIR="output/ecdysis/morphospecies"
DATASET_NAME="$1"
PREVIOUS_VERSION="$2"
THIS_VERSION="$3"

export GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "This new version is: ${THIS_VERSION}"
echo "Found ${GPU_COUNT} GPU(s)"

# Copy dataset report from dataset to model folder
. ./deploy/copy_reports.sh "${DATASET_NAME}" "${OUTPUT_DIR}" "${THIS_VERSION}"

# Run training starting from last best checkpoint
/home/ecdysis/miniconda3/envs/pytorch/bin/torchrun --nproc_per_node ${GPU_COUNT} main.py --cfg configs/ecdysis.yaml \
 --data-path "datasets/${DATASET_NAME}/" --tag "$1" --version "$THIS_VERSION" \
  --pretrain "${OUTPUT_DIR}/${STARTING_CHECKPOINT}/best.pth" >/dev/null
wait
# Evaluate trained model
/home/ecdysis/miniconda3/envs/pytorch/bin/torchrun --nproc_per_node ${GPU_COUNT} main.py \
  --cfg "${OUTPUT_DIR}/${THIS_VERSION}/config.yaml" --dataset bugbox --data-path "datasets/${DATASET_NAME}" --eval  --pretrain "${OUTPUT_DIR}/${THIS_VERSION}/best.pth" --version "$THIS_VERSION" > "${OUTPUT_DIR}/${THIS_VERSION}/console_output.txt"
wait
