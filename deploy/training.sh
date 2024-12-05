#!/bin/bash

# This script is used to train a new model and deploy it on the server.
# Arguments: $1: deployment server address, $2: model tag

set -eE  # Exit if any command fails https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/


cd /home/ecdysis/MetaFormer/

MODEL_PREFIX="output/ecdysis/$1"


DEPLOYED_VERSION=$(curl  "$1:8085/models/metaformer" -s | jq -r .[0].modelVersion)
VERSION_MAJOR=$(echo "$LAST_VERSION" | cut -d. -f1)
VERSION_MINOR=$(echo "$LAST_VERSION" | cut -d. -f2)
THIS_VERSION="test$2"

export GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Last version: 1.22" #${LAST_VERSION}"
echo "This new version is: ${THIS_VERSION}"
echo "Found ${GPU_COUNT} GPU(s)"

# Backup old model and dataset
DATASET="bugbox_model_${THIS_VERSION}"

python -m dataset_generation --dataset-name "$DATASET" --train-size 0.8 --minimum-images 20 --drop-duplicates
wait

# Copy dataset report from dataset to model folder
. ./deploy/copy_reports.sh "${DATASET}" "${MODEL_PREFIX}" "${THIS_VERSION}"


# Run training starting from last best checkpoint
python -m torch.distributed.launch --nproc_per_node ${GPU_COUNT} --master_port 12345 main.py --cfg configs/ecdysis_test.yaml \
 --data-path "datasets/${DATASET}/" --tag "$1" --version "$THIS_VERSION" \
  --pretrain "output/ecdysis/morphospecies/1.22/best.pth" --ignore-user-warnings # > "${MODEL_PREFIX}/${THIS_VERSION}/console_output.txt" # Avoid user warnings on logs
wait
# Evaluate trained model
python -m torch.distributed.launch --nproc_per_node ${GPU_COUNT} --master_port 12345 main.py \
  --cfg "${MODEL_PREFIX}/${THIS_VERSION}/config.yaml" --dataset bugbox --data-path "datasets/${DATASET}" --eval  --pretrain "${MODEL_PREFIX}/${THIS_VERSION}/best.pth" --version "$THIS_VERSION" > "${MODEL_PREFIX}/${THIS_VERSION}/console_output.txt"
wait
