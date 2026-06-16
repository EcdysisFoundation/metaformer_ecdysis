#!/bin/bash
# Copy dataset report from dataset to model folder
# Parameters:
# $1: dataset directory
# $2: model PREFIX
# $3: new model VERSION
DATASET=$1
OUTPUT_DIR=$2
THIS_VERSION=$3
# As training can take long it's better to avoid referencing the dataset folder, put all required files in the model folder
mkdir -p "${OUTPUT_DIR}/${THIS_VERSION}"

cp "datasets/${DATASET}/dataset_report.csv" "${OUTPUT_DIR}/${THIS_VERSION}/dataset_report.csv"
cp "datasets/${DATASET}/morphospecies_map.csv" "${OUTPUT_DIR}/${THIS_VERSION}/morphospecies_map.csv"
cp "dataset_generation/training_selections.csv" "${OUTPUT_DIR}/${THIS_VERSION}/training_selections.csv"
