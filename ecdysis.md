> *Note: All commands assume the working directory is the root of this repository*

## Environment

All required packages are installed in the `metaformer` conda virtual environment. To activate it use `conda activate
metaformer` inside the shell. `conda run -n metaformer <command>` can be used to run a command inside the environment 
without changing the base environment.

The metaformer environment is the one that currently works. As far as I can tell, at present the metaformer-amp environment contains package versions that do not work with the model. Attempting to use Nano in the metaformer environment results in a segmentation fault, so I've been using vim.

## Dataset generation

Test and training CSV files are generated on ecdysis01 in `/srv/bugbox3/bugbox3/core/management/commands/create_test_csv.py` with Django queries. 

To create the file, run `sudo docker compose -f local.yml run --rm django python manage.py create_test_csv` from the bugbox root directory. 
An scp command can be used to copy the file from `/srv/bugbox3/bugbox3/core/management/commands/testing_data` in ecdysis01 into `testing_data/` in the MetaFormer root directory on ecdysis02. 

To generate a dataset ready for model training execute `python -m dataset_generation`. Use the option `-h` to see all
options available. The parameters to set the connection to the remote server should be inside `connection.py` in the
`dataset_generation` directory.



### Scripts and modules
#### `data.py`
Contains a custom class to interact with the csv file generated from BugBox's database.

#### `queries.py`
Contains useful sql queries to extract data from BugBox's database. No longer used and could be deleted.

#### `generate_tree.py`

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
*usage*: 
```
  generate_data_from_csv.py [-h] [--top-hierarchy TOP_HIERARCHY] [--debug] data_path csv_path output_path
```
*positional arguments*:
``` 
  data_path             Path to the input data root directory
  csv_path              Path to the csv file of insect data
  output_path           Output directory path
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

*usage*: 
```
split_insect_samples.py [-h] [--train-size TRAIN_SIZE] [--levels LEVELS] [--min-images MIN_IMAGES] [--debug] 
  [--seed SEED] [--no-copy] [--follow-symlinks] [--yaml-file YAML_FILE] [--add-reference-images]
  [--reference-image-path REFERENCE_IMAGE_PATH] input_directory output_directory
```
*positional arguments*:
```
  input_directory       Path to input directory. Images must be inside subdirectories named as the class they belong to.
  output_directory      Output directory
```


## Training and evaluation

### Main MetaFormer script



To run a training using both GPUs available on a generated dataset execute
```commandline
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main.py --cfg configs/ecdysis.yaml
 --dataset bugbox --data-path datasets/bugbox_morphospecie --tag tag --version version
 --pretrain output/ecdysis/training_dir/last-trained-model.pth --ignore-user-warnings --use-checkpoint
```
The parameters and options for the training procedure are defined in the configuration file (configs/ecdysis.yaml 
in the example). Arguments passed in the command line will overwrite those in the file. 

To evaluate the model on the test set use
```commandline
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main.py --cfg output/ecdysis/model-name/config.yaml
 --dataset bugbox --data-path datasets/bugbox_morphospecie --tag tag --version version
 --pretrain output/ecdysis/model-name/best.pth --ignore-user-warnings --use-checkpoint --eval
```
Make sure to use the correct config and pretrained weights. Classification statistics are saved in csv format inside the
output/ecdysis/model-name/ directory.

### Tensorboard

Training and validation metrics are saved to tensorboard by default. To run the Tensorboard server on a particular model 
execute
```commandline
tensorboard --logdir output/ecdysis/model-name/tensorboard --bind_all
```
then open a browser and go to [](host-or-ip:6006)


## Deployment scripts


### Test model `deploy/training.sh` 
This script creates training set from a local .csv file and trains a MetaFormer model on this data. Does not deploy to the server. 
*usage*:
```commandline
   bash deploy/training.sh "test_directory" "test_model_name"
```
   *positional arguments*:
```
    test_directory       Directory inside the output/ecdysis directory
    test_model_name      Name of the model, will be a directory inside output/ecdysis/test_directory/
 ```

### Test model deployment `deploy/test.sh`

This script automatically lunches a local *Torchserve* instance, publishes the specified model and sends a picture to
the prediction endpoint.
*usage*:
```commandline
    bash deploy/test.sh "model_name" "image/path"
```
*positional arguments*:
```
    model_name        Metaformer model name (directory inside output/MetaFG_2 directory)
    image_path        Path to the image to be send for testing
```

### Serve script `deploy/serve.sh`

Serves model to a running *Torchserve* container with its management API published at port 8085. The archive is forcibly
recreated, so it will overwrite an old archive if the new shares the same name.

*usage*:
```commandline
    bash deploy/serve.sh "model_name" "host_address" "model_version"
```
*positional arguments*:
```
    model_name        Metaformer model name (directory inside output/ecdysis directory)
    host_address      Host name or IP address (i.e. ecdysis01.local, localhost, 127.0.0.1)
    model_version     Version identifier of the model (i.e. 1.0)
```

### Automatic training script `deploy/automatic_training.sh`

Used as a cronjob to perform regular trainings. It automatically generates a dataset with all images available on
BugBox, executes training and evaluation and deploys the model with the next available minor version.

*usage*
```commandline
    bash deploy/automatic_training.sh "host-address" "model_name"
```

*positional_arguments*:
```
    host_address      Host name or IP address (i.e. ecdysis01.local, localhost, 127.0.0.1)
    model_name        Metaformer model name (directory inside output/ecdysis directory)
```

For manually lunched trainings you can also use the command:
```commandline
run-metaformer-training
```
This is an alias for `conda run --live-stream -n metaformer-amp /home/ecdysis/MetaFormer/deploy/automatic_training.sh 
ecdysis01.local morphospecies`.

> *Note:* Using `nohup` will not prevent the training process to be killed when the shell session is terminated. This is
> due to an issue with signaling and Pytorch's distributed training (see https://github.com/pytorch/pytorch/issues/67538).
> To run the training script in the background and close the terminal you must use `tmux` instead.

### Torchserve inference and management

The model are deployed using [Torchserve](https://pytorch.org/serve/). The management API is published at port 8085 and 
the inference API at port 8084.

To check which models are currently deployed use
```commandline
curl host:8085/models
```

To get information about a specific model use
```commandline
curl host:8085/models/model_name
```

To request a classification of a local image use
```commandline
curl -X POST host:8084/predictions/model_name -T path/to/image.jpg
```

