## Environment

All required packages are installed in the `pytorch` conda virtual environment. To activate it use `conda activate pytorch` inside the shell.

To update or install libraries, only use the envionment.yml file. Edit the file and use the following to update the environment. Only install python from Conda Forge, then install all Python packages from PIP because Pytorch is only supported on PIP now, and installing everything through PIP will help ensure library compatibilities.

`conda env update --file environment.yml  --prune`

## Dataset generation

An image selection .csv file identifies the location and classification of images to use. Copy this file to `/dataset_generation/training_selections.csv` . See expected format in `dataset_generation/data.py` We version each training with a name (example DATASET_NAME). This name is used in directory creation.

To initiate dataset generation

`python -m dataset_generation DATASET_NAME --img-mnt /pool1/srv/bugbox3/bugbox3/media/ --train-size 0.8 --minimum-images 20 --drop-duplicates`

### Symlink image files

The images are accessed through symlinks created during dataset generation. The drive location on Ecdysis01 needs to be mapped for this to work. This is done with the following command, and will need to be re-linked after a system reboot.

`sudo sshfs ecdysis@ecdysis01.local:/pool1/srv/bugbox3/bugbox3/media/ /pool1/srv/bugbox3/bugbox3/media/ -o allow_other`

Can check if the entry still exists by viewing filesystem usage with `df -H`


## Training

Currently training is done with ... `deploy/training.sh`. This uses the training selections file to structure images and files for model traning (see `dataset_generation`), starts the training, and runs some analytics at the end. It does not deploy to the trained model to the server. Output should be reviewed before deploying the newly trained model.

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

Deployment start with reviewing the trianing output. Move files to the share drive for review. This assumes DIRECTORY == 'morphospecies'. On Ecdysis01, with the new MODEL_NAME ..

    mkdir /pool1/smb/metaformer-stats/MODEL_NAME

    scp ecdysis@ecdysis02.local:~/MetaFormer/output/ecdysis/morphospecies/MODEL_NAME/dataset_report_stats.csv /pool1/smb/metaformer-stats/MODEL_NAME/dataset_report_stats.csv

    scp ecdysis@ecdysis02.local:~/MetaFormer/output/ecdysis/morphospecies/training_results.csv /pool1/smb/metaformer-stats/MODEL_NAME/training_results.csv


The model is then deployed for inference using the FastAPI app documented here https://github.com/EcdysisFoundation/inference-fastapi


## Tensorboard

Training and validation metrics are saved to a directory called tensorboard. To view these, download the directory, and run the Tensorboard server to view. You will need a local environment with tensorboard installed, then following command where tensorboard is the path to the tensorboard directory.
execute
```commandline
tensorboard --logdir tensorboard --bind_all
```
then open a browser and go to to the local computer url indicated on the command line

## Origination

This repo was forked from https://github.com/dqshuai/MetaFormer
