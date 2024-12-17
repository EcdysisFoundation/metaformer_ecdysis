import csv
import logging
from pathlib import Path
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics import Accuracy, Precision, Recall, F1Score, StatScores, ConfusionMatrix, MetricCollection
from yacs.config import CfgNode

from datetime import datetime

from .dataset_generation.data import BugBoxData
from utils import save_json


def get_model_metrics(config: CfgNode):
    """
    Get metric collection for model training and evaluation
    Args:
        config: Model's configuration file

    Returns: torchmetrics collection
    """
    metrics = [Accuracy(num_classes=config.MODEL.NUM_CLASSES, average='micro'),
               Precision(num_classes=config.MODEL.NUM_CLASSES, average='macro'),
               Recall(num_classes=config.MODEL.NUM_CLASSES, average='macro'),
               F1Score(num_classes=config.MODEL.NUM_CLASSES, average='macro')]

    if config.EVAL_MODE:
        metrics.append(StatScores(num_classes=config.MODEL.NUM_CLASSES, reduce='macro'))
        metrics.append(ConfusionMatrix(num_classes=config.MODEL.NUM_CLASSES))

    return MetricCollection(metrics)

def _get_stats_from_metrics(metrics:MetricCollection,total_column_name:str) -> dict:
    """
    Get per class statistics
    Args:
        metrics: torchmetrics collection, has to have a `StatScores` metric
        total_column_name: name of the field with the total evaluation sample count
    Returns:
        dict[str,numpy array]
    """

    stats = metrics['StatScores']
    tp, fp, tn, fn = stats.tp.cpu().numpy(), stats.fp.cpu().numpy(), stats.tn.cpu().numpy(), stats.fn.cpu().numpy()

    #print('tp' + str(tp))
    #print('fp' + str(fp))
    #print('tn' + str(tn))
    #print('fn' + str(fn))
    return {'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Precision': tp / (tp + fp),
            'Recall': tp / (tp + fn),
            'F1': 2*tp / (2*tp + fp + fn),
            total_column_name: tp + fn
           }

def _check_test_count(row):
    """ Aux method which returns True if test count from the split is different from test count from the evaluation """
    if row["test"] != row["total_test"]:
        logging.error(f"Different test count:\n{dict(row)}")
        return True
    return False

def get_json_stats(metrics: MetricCollection, class_ids: List, version:str,id_name:str, output:Path=None, split_df=None, display=3):
    """
    Get and save per class statistics in JSON format
    Args:
        metrics: torchmetrics collection, has to have a `StatScores` metric
        class_ids: List of class ids in the same order as the dataset
        id_name: Name to use for the class ids
        output: Output JSON path if set
        split_df: Dataframe with the dataset split report

    Returns: Statistics data frame
    """

    pd.set_option('display.max_columns', None)  # show all columns in the head() method

    stats_data = {k.lower():v for k,v in _get_stats_from_metrics(metrics,"total_test").items()}

    stats = pd.DataFrame(data=stats_data).fillna(0)
    pd.set_option('display.max_rows', 2000)

    stats[id_name] =list(class_ids)

    # Debug before and after
    logging.debug(f"Before splits merge:\n{stats.head(display)}")
    if split_df is not None:
        logging.debug(f"Split df:\n{split_df.head(display)}")
        # will merge with the name, for debugging purposes
        split_df = split_df[["morphos_id","morphos_name","train","test","val","total_samples"]]

        # They should have the same classes, but let's use left to catch any issues
        stats = split_df.merge(stats,how="left",right_on="morphos_id",left_on="morphos_id")
    logging.debug(f"After splits merge:\n{stats.head(display)}")

    # verify that total_test == test, as as check.
    if stats.apply(lambda row: _check_test_count(row), axis=1).any():
        logging.error("Different test count, see above.")
    else:
        logging.warning("Expected and actual test count are the same (OK).")

    stats.rename(columns={"total_samples":"total"},inplace=True)
    # drop total_test column (which is only test), the total we want is train+test+val
    stats.drop(["total_test","morphos_name"],inplace=True,axis=1,errors='raise') #remove debug columns

    # orient='records' follows the format [{"precision":0.38,"recall":1,"total":5,"f1":0.56,"morphospeciea_id":10},...]
    # result adds the version field and puts the report inside data
    result = {"version":str(version),"data":stats.to_dict(orient='records')}
    if output:
        # Save the JSON to later send it to the endpoint
        save_json(result,output)
    return result


def get_stats(metrics: MetricCollection, class_names: List[str], output: Path, version: str, save_csv: bool = True):
    """
    Get and save per class statistics
    Args:
        metrics: torchmetrics collection, has to have a `StatScores` metric
        class_names: List of class names
        output_dir: Output directory path
        save_csv: Whether to save as a csv file

    Returns: Statistics data frame
    """
    stats_data = _get_stats_from_metrics(metrics,'Total samples')
    stats = pd.DataFrame(data=stats_data, index=class_names).fillna(0)  # fill NaNs with 0 in case tp + fp = 0

    if save_csv:
        stats.assign(model_name=version)
        morhospecies_df = BugBoxData.get_morhospecies_df()
        morhospecies_df.set_index('morphos_id', inplace=True)
        stats = stats.merge(morhospecies_df, left_index=True, right_on='morphos_id')
        stats.to_csv(output)
    return stats


def plot_confusion_matrix(metrics: MetricCollection, class_names: List[str], output: Path, save: bool = True):
    """
    Render and save confusion matrix
    Args:
        metrics: torchmetrics collection, has to have a `ConfusionMatrix` metric
        class_names: List of class names, sorted as in the model output
        output_dir: Output directory path
        save: Whether to save as a csv file
    """
    matrix = metrics['ConfusionMatrix'].confmat.cpu().numpy()
    matrix_display = ConfusionMatrixDisplay(matrix, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(40, 40))
    matrix_display.plot(xticks_rotation='vertical', colorbar=False, ax=ax)

    if save:
        # confusion_matrix = Path(output_dir) / 'confusion_matrix.png'
        plt.savefig(output)

    plt.show()


def log_metrics(logger: logging.Logger, metrics: MetricCollection, aggregation: str):
    """
    Simple function to log classification metrics at batch or epoch level
    Args:
        logger:
        metrics:
        aggregation:

    Returns:

    """
    metrics_string = ' | '.join(f'{m} = {v}' for m, v in metrics.items() if m not in ['StatScores', 'ConfusionMatrix'])
    logger.info(f'{aggregation.title()} metrics:\n\t{metrics_string}')


def dump_summary(metrics: MetricCollection, config: CfgNode, dump: bool = False):
    """
    Dump summary metrics to a CSV file 'training_results.csv'
    """

    summary = {
        'name': config.TAG,
        'version': config.VERSION,
        'train_images': config.DATA.TRAIN_SAMPLES,
        'test_images': config.DATA.TEST_SAMPLES,
        'number_of_classes': config.MODEL.NUM_CLASSES,
        'accuracy': round(metrics['Accuracy'].item(), 3),
        'precision': round(metrics['Precision'].item(), 3),
        'recall': round(metrics['Recall'].item(), 3),
        'f1': round(metrics['F1Score'].item(), 3),
        # add date
        'date': datetime.now().strftime("%Y-%m-%d")
    }

    if dump:
        csv_file = Path(config.OUTPUT).parent/'training_results.csv'
        exists = csv_file.is_file()
        with open(csv_file, 'a') as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(summary.keys())
            writer.writerow(summary.values())

    return summary
