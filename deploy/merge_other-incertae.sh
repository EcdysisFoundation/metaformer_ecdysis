#!/bin/bash

OTHER_NAME=3443
INCERTAE_SEDIS_NAME=3458

echo "Merging 'other' category into 'incertae sedis'"

rsync -a "datasets/$1/train/$OTHER_NAME/" "datasets/$1/train/$INCERTAE_SEDIS_NAME/" --remove-source-files
rm -r "datasets/$1/train/$OTHER_NAME"
rsync -a "datasets/$1/val/$OTHER_NAME/" "datasets/$1/val/$INCERTAE_SEDIS_NAME/" --remove-source-files
rm -r "datasets/$1/val/$OTHER_NAME"
rsync -a "datasets/$1/test/$OTHER_NAME/" "datasets/$1/test/$INCERTAE_SEDIS_NAME/" --remove-source-files
rm -r "datasets/$1/test/$OTHER_NAME"
