## Data scripts

Run `python -m data_pipeline {GBIF,morphospecie}` to generate a ready to train and evaluate dataset from BugBox images,
using either GBIF's taxon ids or Ecdysis's Morphospecie as classification criteria. Use `-h` option to get information
about other optional parameters.

> Execute all scripts from the root directory of the repository and as a python module, e.g. 
> `python -m data_scripts.generate_tree ...`. Use `-h` option to get usage instructions.

### `generate_tree.py`

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
*usage*: generate_data_from_csv.py [-h] [--top-hierarchy TOP_HIERARCHY] [--debug] data_path csv_path output_path
*positional arguments*:
  data_path             Path to the input data root directory
  csv_path              Path to the csv file of insect data
  output_path           Output directory path

### `split.py`

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

*usage*: split_insect_samples.py  [-h] [--train-size TRAIN_SIZE] [--levels LEVELS] [--min-images MIN_IMAGES] 
                                [--debug] [--seed SEED] [--no-copy] [--follow-symlinks]
                                [--yaml-file YAML_FILE] [--add-reference-images] [--reference-image-path REFERENCE_IMAGE_PATH]
                                input_directory output_directory

*positional arguments*:
  input_directory       Path to input directory. Images must be inside subdirectories named as the class they belong to.
  output_directory      Output directory

### `db.py`

Has a custom class with useful methods to connect to BugBox database and retrieve data. It is not ment to be used as a
script. Connection parameters should be added to an `connection.py` file.

### `queries.py`

SQL queries templates used to get images metadata form BugBox database.

