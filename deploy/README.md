## Automatic training and deployment

### Torchserve
We use [Torchserve](https://pytorch.org/serve/) for model deployment. Before deploying, the model definition and weights 
need to be archived in a `.mar` file. Optionally, a file containing the mapping between the model's output indices and
the classes names can be provided.

To create a model archive ready to be deployed to Torchserve, run the following command:

```bash
torch-model-archiver --model-name metaformer --version 1.0 --model-file models/MetaFG.py \
  --serialized-file MODEL/PATH/best.pth --handler deploy/handler.py \
  --export-path deploy/model_store/ --requirements-file deploy/requirements.txt \
  --extra-files "config.py,MODEL/PATH/config.yaml,models/,deploy/inference.py,deploy/taxon_map.csv"
```

Specific requirements can be added via the `--requirements-file` argument. Note that to use this option the 
`install_py_dep_per_model` parameter has to be set to `true` in the server (the default is `false`).

### Inference
The `inference.py` module has the base class for inference with the `metaformer` model. This module is used in the 
deployment but can be use independently, for example to test inference on a single image.

### Handler
`handler.py` contains a subclass of Torchserve's `VisionHandler`. It defines the handling of inputs and the format of 
the response for the model.

### Automatic training
The bash script `automatic_training.sh` performs all the steps to train a new model and deploy it to a running 
Torchserve service.

1. Backs up the last dataset and model.
2. Generates a new dataset from BugBox images.
3. Runs training using the new generated dataset.
4. Evaluates performance on the test split.
5. Serves the model calling `serve.sh`.
6. Syncs the output stats files to `pool1/smb` shared directory. This includes the evaluation result, the counts per split and class and which classes were below the threshold.
7. Sends a JSON containing the data from the previous per-class reports to an endpoint in the following format:
```json
{
  "version": "1.18",
  "data": [
    {
      "train": int,
      "test": int,
      "val": int,
      "total": int,
      "tp": int,
      "fp": int,
      "tn": int,
      "fn": int,
      "precision": float,
      "recall": float,
      "f1": float,
      "morphospecie_id": int      
    },
    ...
  ]
}
```
Where train, test, and val are the counts for that morphospecie for each split, tp,fp,tn,fn,precision, recall and f1 are the results of the evaluation, and total=train+test+val.

7. Compresses old data to save storage space

To execute the script you need to provide an address to the host running the Torchserve service (inference port 8084 and
8085 for management) and a name for the new model. For example, if the following is executed:

```commandline
bash deploy/automatic_training localhost morphospecies
```

the training outputs will be stored in `./outouts/ecdysis/morphospecies` and the dataset in 
`./datasets/bugbox_morphospecies`. The trained model will be published in the localhost Torchserve running instance.

#### Test only training

The script `test_training.sh` is a reduced version of `automatic_training.sh` that only runs the training/evaluation part to test functionality without performing any deployment. This expects a ecdysis_test.yaml file in the configs directory, that runs a single epoch. Since no deployment is performed the first parameter (the host name) is ignored.
It will add a model and an entry to the train_results.csv in the selected directory.

```commandline
bash deploy/automatic_training localhost test_trainings
```

Likewise the script `test_naming.sh` with the same parameters only runs the version naming part



#### Serve a model
The `serve.sh` script archives and publish a model to a running Torchserve service. It is used inside 
`automatic_training.sh`. Usage:

```commandline
bash deploy/serve.sh morphospecies localhost 1.0
```

1. Archives the model.
2. Copies the archive to `/pool1/model-store` shared directory.
3. Publishes the model on running torchserve service.
4. Set the number of workers and the default version.
5. Test the process was succesful by asking for a prediction of a known easy image.

*Note:* The archiver will overwrite the old file if the new one has the same name.

#### Test deployment
`test.sh` test the deployment by creating a local instance of Torchserve and testing an image. After the test, the
service is killed. This script is useful while developing a handler or changing the base model.