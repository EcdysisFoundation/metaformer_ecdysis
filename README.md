# metaformer_ecdysis

This repo uses a MetaFormer modeling approach to serve as an AI classification model for the Ecdsysis Foundation's ( https://www.ecdysis.bio/ ) BugBox applcation, see https://bugbox.ecdysis.bio/ . As part of our work, we generate biological inventories of farm sites in various stages of regenerative adoption. One aspect of this is collecting and identifying many hundreds of thousands of insects. We use artificial intelligence to give preliminary identifications to each insect we collect, then review a proportion of identifications and retrain our model to improve it over time. This activity is described in the following peer review publication.

Welch, K. Wilson, M. Lundgren, J (2025). Evaluation of BugBox, a software platform for AI-assisted bioinventories of
arthropods. Journal of Animal Ecology. https://doi.org/10.1111/1365-2656.70178

## Origination

This repo was originally forked from https://github.com/dqshuai/MetaFormer . A part of this code was borrowed from https://github.com/microsoft/Swin-Transformer

The MetaFormer model is described in the original README here https://github.com/EcdysisFoundation/metaformer_ecdysis/blob/main/metaformer.md

Updates to these include compatibility to torch==2.6.0.

## Environment

All required packages are installed in the `pytorch` conda virtual environment, see `environment.yml`. See `environment_freeze.yml` for exact versions last confirmed to work with our Ubuntu operating system, Intel i9 CPU, and two Nvidia RTX GPUs.

## Dataset generation

An image selection .csv file identifies the location and classification of images to use. Copy this file to `/dataset_generation/training_selections.csv` . We use this process to create symlinks to another location on a local network. See expected format in `dataset_generation/data.py` DATASET_NAME is used in directory creation.

To initiate dataset generation

`python -m dataset_generation DATASET_NAME --img-mnt /pool1/srv/bugbox3/bugbox3/media/ --train-size 0.8 --minimum-images 20`


## Training

Training is initiated with ... `deploy/training.sh`. This uses the image directories sctructured from `dataset_generation`, starts the training, and runs some analytics at the end.

We version each training with a name (THIS_VERSION). This can be the same as the DATASET_NAME or if multiple trainings are run on the same dataset, varying them with a diffent value for THIS_VERSION keeps from overwriting previous runs. This name is used in directory creation.

*usage*:
```commandline
    conda activate pytorch

    bash deploy/training.sh DATASET_NAME PREVIOUS_VERSION THIS_VERSION
```
   *positional arguments*:

    - DATASET_NAME      Name of the dataset directory.
    - PREVIOUS_VERSION  The directory name of the previous best.pth checkpoint
    - THIS_VERSION      Names a new directory inside OUTPUT_DIR and is used as the model version in the inference response

 Traing can be ran with one or two epochs for testing (for example using config `configs/ecdysis_test.yaml`). To run in background and write output to a file, append the following to the command above ` > file.log 2>&1 &`.

## Deployment

The model is then deployed for inference using the FastAPI app documented here https://github.com/EcdysisFoundation/inference-fastapi

## Tensorboard

Training and validation metrics are saved to a directory called tensorboard. To view these, download the directory, and run the Tensorboard server to view. You will need a local environment with tensorboard installed, then following command where tensorboard is the path to the tensorboard directory.
execute
```commandline
tensorboard --logdir tensorboard --bind_all
```
then open a browser and go to to the local computer url indicated on the command line
