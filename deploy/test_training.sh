#!/bin/bash

# This script is used to train a new model and deploy it on the server.
# Arguments: $1: deployment server address, $2: model tag

set -eE  # Exit if any command fails https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/


cd /home/ecdysis/MetaFormer/
# Mount directories if needed
bash deploy/mount_dirs.sh || true

MODEL_PREFIX="output/ecdysis/$2"


DEPLOYED_VERSION=$(curl  "$1:8085/models/metaformer" -s | jq -r .[0].modelVersion)
LAST_VERSION=$(find "output/ecdysis/$2" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" | sort -V -r | head -n 1)
VERSION_MAJOR=$(echo "$LAST_VERSION" | cut -d. -f1)
VERSION_MINOR=$(echo "$LAST_VERSION" | cut -d. -f2)
THIS_VERSION=$VERSION_MAJOR.$((VERSION_MINOR + 1))

export GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Last version: ${LAST_VERSION}"
echo "This new version is: ${THIS_VERSION}"
echo "Found ${GPU_COUNT} GPU(s)"

# Backup old model and dataset
#BACKUP="$2_backup_$(date +%Y%m%d)"
#mv "${MODEL_PREFIX}/$2" "${MODEL_PREFIX}/${BACKUP}" || echo "Model does not exist, skipping backup"
DATASET="bugbox_$2"
DATASET_BACKUP="datasets/${DATASET}_backup_$(date +%Y%m%d_%H%M%S)"
mv "datasets/${DATASET}" "${DATASET_BACKUP}" || echo "Dataset does not exist, nothing to backup"


# Update datasaet
python -m dataset_generation morphospecie --dataset-name "$DATASET" --train-size 0.8 --minimum-images 20 --drop-duplicates
wait

# Copy dataset report from dataset to model folder
. ./deploy/copy_reports.sh "${DATASET}" "${MODEL_PREFIX}" "${THIS_VERSION}"

# Run training starting from last best checkpoint
python -m torch.distributed.launch --nproc_per_node $GPU_COUNT --master_port 12345 main.py --cfg configs/ecdysis_test.yaml \
   --data-path "datasets/${DATASET}/" --tag "$2" --version "$THIS_VERSION" \
   --pretrain "${MODEL_PREFIX}/${LAST_VERSION}/best.pth" --ignore-user-warnings  # Avoid user warnings on logs
wait

# Evaluate trained model
python -m torch.distributed.launch --nproc_per_node $GPU_COUNT --master_port 12345 main.py \
  --cfg "output/ecdysis/test_trainings/23/config.yaml" --dataset bugbox --data-path "datasets/${DATASET}" --eval \
  --pretrain "${MODEL_PREFIX}/${THIS_VERSION}/best.pth" --version "$THIS_VERSION"
wait

