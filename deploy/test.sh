#!/bin/bash

OUTPUT_PATH=output/MetaFG_2/bugbox+ref_genera_384_4xlr

echo "Archiving model"
torch-model-archiver --model-name metaformer --version 1.0 --model-file models/MetaFG.py \
  --serialized-file $OUTPUT_PATH/best.pth --handler deploy/MetaFG_2/handler.py \
  --export-path deploy/MetaFG_2/model_store/ \
  --extra-files config.py,$OUTPUT_PATH/config.yaml,models/,$OUTPUT_PATH/taxon_map.csv \
  --force
echo "Archiving finished"

echo "Starting server"
torchserve --start --model-store deploy/MetaFG_2/model_store --models metaformer=metaformer.mar --ncs
sleep 20

echo "Sending test image to prediction endpoint"
curl localhost:8080/predictions/metaformer \
  -T datasets/insectfam/val/Coleoptera_Staphylinidae/Coleoptera_Staphylinidae\ \(4\).jpg

echo "Stopping the server"
torchserve --stop

echo "All done"