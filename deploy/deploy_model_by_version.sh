#!/bin/bash

# Deploy specified version of the metaformer model trained on morphospecies.
# The model checkpoint needs to be inside the output directory and compressed.

# Usage: ./deploy_model_by_version.sh <model_tag> <server_address> <model_version>

set -e

DEPLOYED_VERSION=$(curl  "$2:8085/models/metaformer" -s | jq -r .[0].modelVersion || echo "None")
if [[ "$DEPLOYED_VERSION" == "$3" ]]; then
    echo "Model version $3 already deployed"
    exit 0
fi

cd /home/ecdysis/MetaFormer/
MODEL_PREFIX="output/ecdysis/$1"

# Check if the model checkpoint exists
CHECKPOINT="${MODEL_PREFIX}/$3.tar.gz"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Model checkpoint $CHECKPOINT not found. This are the available checkpoints:"
    find "$MODEL_PREFIX" -maxdepth 1 -type f -name "*.tar.gz" -printf "%f, "
    exit 1
else
    echo "Deploying $1 model version $3"
    tar -xzf "${MODEL_PREFIX}/$3.tar.gz"
    trap 'rm -r ${MODEL_PREFIX:?}/$3' EXIT  # Remove the extracted folder on exit
    . ./deploy/serve.sh "$1" "$2" "$3"
    wait
    curl -X DELETE -s "$2:8085/models/metaformer/$DEPLOYED_VERSION" | jq -r .status || echo "No model to remove."
    sleep 2
    echo "Model $1 version $3 deployed"
fi