#!/bin/bash

# Parameters:
# $1: model TAG
# $2: model VERSION

MODEL_PREFIX="output/ecdysis/$1"
DST="/pool1/smb/metaformer-stats"

rsync -az "${MODEL_PREFIX}/training_results.csv" $DST

mkdir "$DST/$2" || true  # ignore error if dir exists
cp "${MODEL_PREFIX}/$2/stats_$2.csv" "${DST}/$2/"
cp "${MODEL_PREFIX}/$2/cmatrix_$2.png" "${DST}/$2/"