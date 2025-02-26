> *Note: All commands assume the working directory is the root of this repository*

## Environment

All required packages are installed in the `pytorch` conda virtual environment. To activate it use `conda activate pytorch` inside the shell.

To update or install libraries, only use the envionment.yml file. Edit the file and use the following to update the environment. Only install python from Conda Forge, then install all Python packages from PIP because Pytorch is only supported on PIP now, and installing everything through PIP will help ensure library compatibilities.

`conda env update --file environment.yml  --prune`

## Dataset generation

Image selection CSV files are generated on Ecdysis01 in `/srv/bugbox3/bugbox3/core/management/commands/...`. Production images are uploaded to AWS S3, while training images are accessed on the Ecdysis01 hardrive. Therefore, new images on AWS S3 need to be synced to Ecdysis01 before training begins. This sync process should be scheduled through BugBox's Celery Beat schedule. Before training a new model, ensure the desired training selections are generated from the BugBox managment command and AWS S3 images are synced after the trainging selections .csv file has been generated.

### Symlink image files

The images are accessed through symlinks created during dataset generation. The drive location on Ecdysis01 needs to be mapped for this to work. This is done with the following command, and will need to be re-linked after a system reboot.

`sudo sshfs ecdysis@ecdysis01.local:/pool1/srv/bugbox3/bugbox3/media/ /pool1/srv/bugbox3/bugbox3/media/ -o allow_other`

Can check if the entry still exists by viewing `proc/self/mounts` as seen below. Or on filesystem usage with `df -H`

`ecdysis@ecdysis01.local:/pool1/srv/bugbox3/bugbox3/media/ /pool1/srv/bugbox3/bugbox3/media fuse.sshfs rw,nosuid,nodev,relatime,user_id=0,group_id=0,allow_other 0 0`

## Training

Currently training is done with ... `deploy/training.sh`. This uses the training selections file to structure images and files for model traning (see `dataset_generation`), starts the training, and runs some analytics at the end. It does not deploy to the trained model to the server. Output should be reviewed before deploying the newly trained model.

*usage*:
```commandline
    conda activate metaformer

    bash deploy/training.sh "DIRECTORY" "MODEL_NAME"
```
   *positional arguments*:

    - DIRECTORY       Directory inside the output/ecdysis directory. Usually 'morphospecies'.
    - MODEL_NAME      Name of the model, will be a directory inside output/ecdysis/test_directory/

 When running with one or two epochs for testing (for example using config `configs/ecdysis_test.yaml`), it can run as above to see output, but for longer runs the terminal will eventually close on its own halting the job. Alternatively, running with `nohup` and running in background with `&` cannot be used, becuse of a bug in older version of Torch that conflicts with nohup. It will also terminate. To run in background and write output to a file, use `> file.log 2>&1 &` as described below. After starting, exiting the terminal session with `exit` so that the ternimal does not terminate with SIGHUP. Some discussion about this issue is here https://discuss.pytorch.org/t/ddp-error-torch-distributed-elastic-agent-server-api-received-1-death-signal-shutting-down-workers/135720   and here https://github.com/pytorch/pytorch/issues/76894 . Alternatively, tmux could be used, see https://github.com/tmux/tmux/wiki.


 In the example below, MODEL_NAME == modelVersion in the inference response. The current protocol is to use text based integer version sequence, starting with 1, 2, 3, 4, ... Past versioning used 1.20, 1.21, 1,22, ending with 1.22.

    conda activate metaformer

    bash deploy/training.sh morphospecies MODEL_NAME > output/ecdysis/morphospecies/last_training.log 2>&1 &

    exit

Then one can return later and determine if it still running with the last_training.log. Once the training is completed, the output, stats, and log should be reviewed to determine if it completed successfully and is suitable to deploy.

## Deployment

Deployment start with reviewing the trianing output. Move files to the share drive for review. This assumes DIRECTORY == 'morphospecies'. On Ecdysis01, with the new MODEL_NAME ..

    mkdir /pool1/smb/metaformer-stats/MODEL_NAME

    scp ecdysis@ecdysis02.local:~/MetaFormer/output/ecdysis/morphospecies/MODEL_NAME/stats.csv /pool1/smb/metaformer-stats/MODEL_NAME/stats.csv

    scp ecdysis@ecdysis02.local:~/MetaFormer/output/ecdysis/morphospecies/MODEL_NAME/dataset_report.csv /pool1/smb/metaformer-stats/MODEL_NAME/dataset_report.csv

    scp ecdysis@ecdysis02.local:~/MetaFormer/output/ecdysis/morphospecies/training_results.csv /pool1/smb/metaformer-stats/MODEL_NAME/training_results.csv

Once review is completed, run the torch-model-archiver on Ecdysis02, in the conda environment, 'metaformer'. Note: this will overwrite deploy/model-store/metaformer.mar.

Save a copy of metaformer.mar in case we need to revert, (see Archive and Revert)

    cp model_store/metaformer.mar output/ecdysis/morphospecies/metaformer_MODEL_NAME.mar

Activate environment and run torch-model-archiver.

    conda activate metaformer

    torch-model-archiver --model-name "metaformer" --version "MODEL_NAME" --model-file "models/MetaFG.py" --serialized-file "output/ecdysis/morphospecies/MODEL_NAME/best.pth" --handler "deploy/handler.py" --export-path "deploy/model_store/" --requirements-file "deploy/requirements.txt" --extra-files "config.py,output/ecdysis/morphospecies/MODEL_NAME/config.yaml,models/,deploy/inference.py,deploy/morphospecies_map.csv" --force

If completed successfully, download this file from Ecdysis01, overwriting the copy of it there.

    scp ecdysis@ecdysis02.local:~/MetaFormer/deploy/model_store/metaformer.mar /pool1/model-store-2/metaformer.mar

Once the new model is served through Torchserve (see below), we update the model statistics in the BugBox database. We will need the dataset_report_stats.csv file to do that, which is dataset_report.csv and stats.csv combined. So download it as well to insert with the appropriate django management command.

    scp ecdysis@ecdysis02.local:~/MetaFormer/output/ecdysis/morphospecies/MODEL_NAME/dataset_report_stats.csv /pool1/srv/bugbox3/local_files/dataset_report_stats.csv

### Archive and Revert

The metaformer.mar file should be saved before overwriting to be able to easily revert to a previous model if necessary. Formerly, only the model output and model dataset files were saved. To recreate a .mar file to revert a deployment could be incumbered by changes in the codebase. Both can be saved for some time, then deleted as we need space to be recovered. More storage is available on Ecdysis01 as well. Here is more info about doing archiving previous models and managing space.

On Ecdysis02 in MetaFormer/output/ecdysis/morphospecies, gzip old model version output folders.

    tar --remove-files -zcvf model_folder.tar.gz MODEL_NAME

Copy metaformer.mar files before overwriting them.

    cp model_store/metaformer.mar output/ecdysis/morphospecies/metaformer_MODEL_NAME.mar

In datasets, ls files and delete some older ones with `rm -r FILENAME`. The same can be done in output/ecdysis/morphospecies, but these are higher priority than the files in datasets. Some stats to inform these decisions are found by below.

Get total directory size of MetaFormer dir,

    du -sh MetaFormer

Get total disk usage, including mount drive

    df -H

### Torchserve inference and management

Once all the metaformer.mar file is generated and moved to model-store-2 on Ecdysis01 as described above, the process can be continued using instructions on the servemetaformer repo. The model are deployed using [Torchserve](https://pytorch.org/serve/). see https://github.com/EcdysisFoundation/servemetaformer


## Scripts and modules

#### `dataset_generation/data.py`
Make a Pandas dataframe from the csv file generated from BugBox's database.


#### `split.py`

Generates train/test/val splits from a directory of classified insect pictures. The output directory structure follows
Imagenet format. Example:
```
root/
├── meta.yaml
├── test
│    ├── Eribolus
│    ├── Liohippelates
│    └── Oscinella
├── train
│    ├── Eribolus
│    ├── Liohippelates
│    └── Oscinella
└── val
     ├── Eribolus
     ├── Liohippelates
     └── Oscinella
```

### Tensorboard

Training and validation metrics are saved to tensorboard by default. To run the Tensorboard server on a particular model
execute
```commandline
tensorboard --logdir output/ecdysis/model-name/tensorboard --bind_all
```
then open a browser and go to [](host-or-ip:6006)
