#!/bin/bash

MODEL_PATH=output/MetaFG_2/$1
IMAGE_PATH=${2:-datasets/insectfam/val/Coleoptera_Staphylinidae/Coleoptera_Staphylinidae\ \(4\).jpg}

echo "Archiving model"
torch-model-archiver --model-name metaformer --version 1.0 --model-file models/MetaFG.py \
  --serialized-file "$MODEL_PATH/best.pth" --handler deploy/handler.py \
  --export-path deploy/MetaFG_2/model_store/ \
  --extra-files "config.py,$MODEL_PATH/config.yaml,models/,$MODEL_PATH/taxon_map.csv" \
  --force
echo "Archiving finished"

echo "Starting server"
torchserve --start --model-store deploy/MetaFG_2/model_store --models metaformer=metaformer.mar --ncs
sleep 20

echo "Sending test image to prediction endpoint"
curl localhost:8080/predictions/metaformer -s \
  -T "$IMAGE_PATH"

echo "Stopping the server"
torchserve --stop

echo "All done"