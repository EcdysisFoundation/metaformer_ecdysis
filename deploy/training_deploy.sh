#!/bin/bash

# This script is used to deploy the model trained with training.sh.

DEPLOYED_VERSION=$(curl "ecdysis01.local:8075/models/metaformer" -H "Accept: application/json" | jq '.[0].modelVersion')
MODEL_PREFIX="output/ecdysis/$1"
SERVERADDRESS="ecdysis01.local"

# Serve new model
if . ./deploy/serve.sh "$1" "$2"; then
  echo "Model version $2 deployed, unregistering old model..."
  curl -X DELETE -s "$SERVERADDRESS:8075/models/metaformer/$DEPLOYED_VERSION" | jq -r .status
else
  echo "Failed to deploy new model"
  exit 1
fi

echo "Syncing stats files..."
. ./deploy/sync_results.sh "$1" "$2"

# Compress backup of old model
echo "Compressing old model..."
tar --remove-files -zcvf "$MODEL_PREFIX/$DEPLOYED_VERSION.tar.gz" "$MODEL_PREFIX/${DEPLOYED_VERSION}" || echo "Model backup does not exist, skipping compression"
