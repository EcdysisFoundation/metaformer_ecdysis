#!/bin/bash

# Copy dataset report from dataset to model folder
# Parameters:
# $1: dataset directory
# $2: model PREFIX
# $3: new model VERSION

DATASET=$1
MODEL_PREFIX=$2
THIS_VERSION=$3

# As training can take long it's better to avoid referencing the dataset folder, put all required files in the model folder
mkdir -p "${MODEL_PREFIX}/${THIS_VERSION}"
# add version number to files
cp "datasets/${DATASET}/dataset_report.csv" "${MODEL_PREFIX}/${THIS_VERSION}/dataset_report_${THIS_VERSION}.csv"
#cp "datasets/${DATASET}/underrepresented_classes.csv" "${MODEL_PREFIX}/${THIS_VERSION}/underrepresented_classes_${THIS_VERSION}.csv"
