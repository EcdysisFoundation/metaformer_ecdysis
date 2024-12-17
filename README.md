> *Note: All commands assume the working directory is the root of this repository*

## Environment

All required packages are installed in the `metaformer` conda virtual environment. To activate it use `conda activate metaformer` inside the shell.

At present the metaformer-amp environment contains package versions that do not work with the model. Attempting to use Nano in the metaformer environment results in a segmentation fault, but vim works. Nano in metaformer environment likely broken from previous modifications of LD_LIBRARY_PATH.

## Dataset generation

Image selection CSV files are generated on Ecdysis01 in `/srv/bugbox3/bugbox3/core/management/commands/...`. Production images are uploaded to AWS S3, while training images are accessed on the Ecdysis01 hardrive. Therefore, new images on AWS S3 need to be synced to Ecdysis01 before training begins. This sync process should be scheduled through BugBox's Celery Beat schedule. Before training a new model, ensure the desired training selections are generated from the BugBox managment command and AWS S3 images are synced after the trainging selections .csv file has been generated.

## Training

Currently training is done with ... `deploy/training.sh`. This uses the training selections file to structure images and files for model traning (see `dataset_generation`), starts the training, and runs some analytics at the end. It does not deploy to the trained model to the server. Output should be reviewed before deploying the newly trained model.

*usage*:
```commandline
    conda activate metaformer

    bash deploy/training.sh "directory" "model_name"
```
   *positional arguments*:

    - directory       Directory inside the output/ecdysis directory
    - model_name      Name of the model, will be a directory inside output/ecdysis/test_directory/

 When running with one or two epochs for testing (for example using config `configs/ecdysis_test.yaml`), it can run as above to see output, but for longer runs the terminal will eventually close on its own halting the job. Alternatively, running with `nohup` and running in background `&` cannot be used, becuse of a bug in older version of Torch that conflicts with nohup. It will also terminate. To run in background and write output to a file. Alternatively, tmux could be used, see https://github.com/tmux/tmux/wiki

    conda activate metaformer

    bash deploy/training.sh morphospecies model_name > output/ecdysis/morphospecies/last_training.log 2>&1 &

    exit

Then one can return later and determine if it still running with the last_training.log


### Scripts and modules
#### `dataset_generation/data.py`
Make a Pandas dataframe from the csv file generated from BugBox's database.

#### `dataset_generation/generate_tree.py` DEPRICATED?

Generates directory structure for classified insect pictures, where every taxon level is represented by a subdirectory.
For example:
```
root/
└── Thysanoptera
    └── Aeolothripidae
        ├── 82773c1e-f540-4f75-9a31-f0e768225cb9_3309.jpg
        ├── 891fbd4a-21fe-406d-a5aa-d80000103141_5578.jpg
        ├── 979f8f3d-5198-4c4f-b424-77bab7d347c5_533.jpg
        └── Aeolothrips
            └── fe7c2bc2-4a66-4826-adf3-bd14c4547229_6739.jpg
```

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


## Deployment scripts

TBD

### Torchserve inference and management

The model are deployed using [Torchserve](https://pytorch.org/serve/). see https://github.com/EcdysisFoundation/servemetaformer
