#!/bin/bash

# This script is used to train a new model and deploy it on the server.
# Arguments: $1: deployment server address, $2: model tag

set -eE  # Exit if any command fails https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/

# Cleanup on error and send failed signal
function failed () {
  echo "The script failed, reverting changes..."
#  mv "${MODEL_PREFIX}/${BACKUP}" "${MODEL_PREFIX}/$1" || echo "Reverting model failed"
  rm -r "datasets/${DATASET}"
  echo "Renaming datasets/${DATASET}_backup_$(date +%Y%m%d) to datasets/${DATASET}" 
  mv "datasets/${DATASET}_backup_$(date +%Y%m%d)" "datasets/${DATASET}" || true
  tar --remove-files -zcvf "${MODEL_PREFIX}/${THIS_VERSION}_failed.tar.gz" "${MODEL_PREFIX}/${THIS_VERSION}" || true
  curl -s "$MONITOR?state=fail"
}

cd /home/ecdysis/MetaFormer/
# Mount directories if needed
bash deploy/mount_dirs.sh || true

MONITOR="https://cronitor.link/p/29925306f8d947d4a1659c63083bb7c1/i5A0js"
EVAL_POST_URL="$1:8000/ai/"
MODEL_PREFIX="output/ecdysis/$2"

# Send started signal
curl -s "$MONITOR?state=run"

DEPLOYED_VERSION=$(curl  "$1:8085/models/metaformer" -s | jq -r .[0].modelVersion)
# sort versions numerically
LAST_VERSION=$(find "output/ecdysis/$2" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" | sort -V -r | head -n 1)
VERSION_MAJOR=$(echo "$LAST_VERSION" | cut -d. -f1)
VERSION_MINOR=$(echo "$LAST_VERSION" | cut -d. -f2)
THIS_VERSION=$VERSION_MAJOR.$((VERSION_MINOR + 1))
echo "Last version: ${LAST_VERSION}"
echo "This new version is: ${THIS_VERSION}"

export GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found ${GPU_COUNT} GPU(s)"

# Backup old model and dataset
DATASET="bugbox_$2"
mv "datasets/${DATASET}" "datasets/${DATASET}_backup_$(date +%Y%m%d)" || echo "Dataset does not exist, nothing to backup"

trap failed ERR

# Update datasaet
python -m dataset_generation morphospecie --dataset-name "$DATASET" --train-size 0.8 --minimum-images 20 --drop-duplicates
wait
#. ./deploy/merge_other-incertae.sh "$DATASET"  # Needed because of the change: other -> incertae sedis
curl -s "$MONITOR?state=ok&msg=Dataset%20generated"

# Copy dataset report from dataset to model folder
. ./deploy/copy_reports.sh "${DATASET}" "${MODEL_PREFIX}" "${THIS_VERSION}"

# Run training starting from last best checkpoint
python -m torch.distributed.launch --nproc_per_node $GPU_COUNT --master_port 12345 main.py --cfg configs/ecdysis.yaml \
   --data-path "datasets/${DATASET}/" --tag "$2" --version "$THIS_VERSION" \
   --pretrain "${MODEL_PREFIX}/${LAST_VERSION}/best.pth" --ignore-user-warnings  # Avoid user warnings on logs
wait
curl -s "${MONITOR}?state=ok&msg=Training%20finished"


# Evaluate trained model
python -m torch.distributed.launch --nproc_per_node $GPU_COUNT --master_port 12345 main.py \
  --cfg "${MODEL_PREFIX}/${THIS_VERSION}/config.yaml" --dataset bugbox --data-path "datasets/${DATASET}" --version "$THIS_VERSION" --eval \
  --pretrain "${MODEL_PREFIX}/${THIS_VERSION}/best.pth"
wait
curl -s "$MONITOR?state=ok&msg=Evaluation%20finished"

# Serve new model
if . ./deploy/serve.sh "$2" "$1" "$THIS_VERSION"; then
  echo "Model version $THIS_VERSION deployed, unregistering old model..."
  curl -X DELETE -s "$1:8085/models/metaformer/$DEPLOYED_VERSION" | jq -r .status
  curl -s "$MONITOR?state=ok&msg=Model%20version%20$THIS_VERSION%20deployed"
else
  echo "Failed to deploy new model"
  curl -s "$MONITOR?state=fail&msg=Failed%20to%20deploy%20new%20model"
  exit 1
fi

echo "Syncing stats files..."
. ./deploy/sync_results.sh "$2" "$THIS_VERSION" "$EVAL_POST_URL"

# Compress backup of old model
echo "Compressing old model..."
tar --remove-files -zcvf "$MODEL_PREFIX/$LAST_VERSION.tar.gz" "$MODEL_PREFIX/${LAST_VERSION}" || echo "Model backup does not exist, skipping compression"
#rm -r "${MODEL_PREFIX:?}/${LAST_VERSION:?}" || echo "Model backup does not exist, skipping deletion"  # SC2115

echo "All steps completed successfully"
curl -s "$MONITOR?state=complete"
