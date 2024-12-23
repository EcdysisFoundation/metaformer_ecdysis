#!/bin/bash

# This script is used to train a new model.

set -eE  # Exit if any command fails https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/


cd /home/ecdysis/MetaFormer/

MODEL_PREFIX="output/ecdysis/$1"


DEPLOYED_VERSION=$(curl "ecdysis01.local:8075/models/metaformer" -H "Accept: application/json" | jq '.[0].modelVersion')
THIS_VERSION="$2"

export GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "deployed version is ${DEPLOYED_VERSION}"
echo "Hardcoded version: 1.22" # replace hardcoding when ready
echo "This new version is: ${THIS_VERSION}"
echo "Found ${GPU_COUNT} GPU(s)"

DATASET="bugbox_model_${THIS_VERSION}"

# data.py may be pointing to the test .csv instead.
echo "Download latest training_selections file."
cp /pool1/srv/bugbox3/local_files/training_selections.csv ./dataset_generation/training_selections.csv || exit 11
wait

python -m dataset_generation "$DATASET" --train-size 0.8 --minimum-images 20 --drop-duplicates
wait

# Copy dataset report from dataset to model folder
. ./deploy/copy_reports.sh "${DATASET}" "${MODEL_PREFIX}" "${THIS_VERSION}"


# Run training starting from last best checkpoint
python -m torch.distributed.launch --nproc_per_node ${GPU_COUNT} --master_port 12345 main.py --cfg configs/ecdysis.yaml \
 --data-path "datasets/${DATASET}/" --tag "$1" --version "$THIS_VERSION" \
  --pretrain "output/ecdysis/morphospecies/1.22/best.pth" --ignore-user-warnings >/dev/null  # only show error messages
wait
# Evaluate trained model
python -m torch.distributed.launch --nproc_per_node ${GPU_COUNT} --master_port 12345 main.py \
  --cfg "${MODEL_PREFIX}/${THIS_VERSION}/config.yaml" --dataset bugbox --data-path "datasets/${DATASET}" --eval  --pretrain "${MODEL_PREFIX}/${THIS_VERSION}/best.pth" --version "$THIS_VERSION" > "${MODEL_PREFIX}/${THIS_VERSION}/console_output.txt"
wait
