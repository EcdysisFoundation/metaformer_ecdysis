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
