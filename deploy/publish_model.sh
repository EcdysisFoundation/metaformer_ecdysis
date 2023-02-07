#!/bin/bash

OUTPUT_PATH="output/MetaFG_2/$1"

curl -X DELETE "http:/ecdysis01.local:8085/models/metaformer"

echo "Archiving model"
torch-model-archiver --model-name "metaformer" --version 1.0 --model-file "models/MetaFG.py" \
  --serialized-file "$OUTPUT_PATH/best.pth" --handler "deploy/MetaFG_2/handler.py" \
  --export-path "deploy/MetaFG_2/model_store/" --requirements-file "deploy/requirements.txt" \
  --extra-files "config.py,$OUTPUT_PATH/config.yaml,models/,$OUTPUT_PATH/taxon_map.csv" \
  --force
echo "Archiving finished"

echo "Sending archived file to ecdysis01"
sftp ecdysis@ecdysis01.local:/pool1/model-store <<< $'put deploy/MetaFG_2/model_store/metaformer.mar'
wait

echo "Publishing model"
curl -X POST "ecdysis01.local:8085/models?url=metaformer.mar&initial_workers=1&synchronous=true"
wait
curl -X PUT ecdysis01.local:8085/models/metaformer?min_worker=1
wait