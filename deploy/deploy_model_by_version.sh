#!/bin/bash

# Deploy specified version of the metaformer model trained on morphospecies.
# The model checkpoint needs to be inside the output directory and compressed.

# Usage: ./deploy_model_by_version.sh <model_tag> <model_version> <server_address>

MODEL_PREFIX="output/ecdysis/$1"

# Check if the model checkpoint exists
CHECKPOINT="${MODEL_PREFIX}/$2.tar.gz"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Model checkpoint $CHECKPOINT not found. This are the available checkpoints:"
    find output/ecdysis/test/ -maxdepth 1 -type f -name "*.tar.gz" -printf "%f, "
    exit 1
else
    DEPLOYED_VERSION=$(curl  "$1:8085/models/metaformer" -s | jq -r .[0].modelVersion)
    echo "Deploying model $1 version $2"
    tar -xzf "${MODEL_PREFIX}/$2.tar.gz"
    . ./deploy/serve.sh "$1" "$3" "$2"
    wait
    rm -r "${MODEL_PREFIX:?}/$2"
    curl -X DELETE -s "$1:8085/models/metaformer/$DEPLOYED_VERSION" | jq -r .status
    sleep 2
    echo "Model $1 version $2 deployed, version $DEPLOYED_VERSION removed."
fi