#!/bin/bash

# This script is used to archive and serve a model on a host running torchserve.
# Arguments: $1: model tag, $2: server address, $3: model version

set -e # Exit if any command fails

MODEL_PATH="output/ecdysis/$1/$3"

echo "Archiving model..."
torch-model-archiver --model-name "metaformer" --version "$3" --model-file "models/MetaFG.py" \
  --serialized-file "$MODEL_PATH/best.pth" --handler "deploy/handler.py" \
  --export-path "deploy/model_store/" --requirements-file "deploy/requirements.txt" \
  --extra-files "config.py,$MODEL_PATH/config.yaml,models/,deploy/inference.py,deploy/taxon_map.csv" \
  --force

echo "Moving archive to pool1..."
cp /home/ecdysis/MetaFormer/deploy/model_store/metaformer.mar /pool1/model-store || exit 11
wait

echo "Publishing model to torchserve..."
ERR_MSG=$(curl -s -X POST "$2:8085/models?url=metaformer.mar&initial_workers=1&synchronous=true" | jq -r '.message')
if [ "$ERR_MSG" != "null" ]
then
  echo "Failed to serve model"
  echo "Response message: $ERR_MSG"
  exit 1
else
  echo "Model published successfully"
fi
wait

curl -X PUT -s "$2:8085/models/metaformer/$3?min_worker=16&max_worker=32" | jq -r .status
curl -X PUT -s "$2:8085/models/metaformer/$3/set-default" | jq -r .status
sleep 30

echo "Sending test request..."
TEST_ID=$(curl -s "$2:8084/predictions/metaformer" -T "tests/diabrotica.JPG" | jq -r .taxonid)
if [ "${TEST_ID}" = 1048497 ]
then
  echo "Test prediction succeeded"
else
  echo "Test prediction failed"
  echo "Test ID: $TEST_ID"
fi
