import csv
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics import Accuracy, Precision, Recall, F1Score, StatScores, ConfusionMatrix, MetricCollection
from yacs.config import CfgNode

from datetime import datetime
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

def get_json_stats(stats_df: pd.DataFrame,id_column:str,version:str,output: Path = None):
    """
    Return and optionally save per class statistics in JSON format with morphospecie id
    Args:
        stats_df: Output of get_stats_with_ids
        id_column: Name of the column to use for the ids
        output: Output json filename path, if not set it will not be saved

    Returns: Statistics data frame with fields: precision, recall, total, f1, id_column (morphospecie_id)
    """
    # get JSON stats using the ids instead of the names
    # all fields in the json are in lower case
    stats_df.rename(columns={c:c.lower() for c in stats_df.columns}, inplace=True)
    # name -> morphospecie_id, total samples -> total
    stats_df.rename(columns={'name':id_column,'total samples':'total'}, inplace=True)
    # reorder columns and keep only the required ones
    stats_df = stats_df[['precision','recall','total','f1',id_column]]
    # orient='records' is [{"precision":0.38,"recall":1,"total":5,"f1":0.56,"morphospecie_id":10},...]
    # result adds the version field and puts the report inside data
    result = {"version":str(version),"data":stats_df.to_dict(orient='records')}
    if output:
        save_json(result,output)
    return result

def get_stats_with_ids(metrics: MetricCollection, class_ids: List):
    """
    Get per class statistics with the list of class_ids instead of the class names
    (This method is separated for testability)
    Args:
        metrics: torchmetrics collection, has to have a `StatScores` metric
        class_ids: List of class or morphospecies ids
    Returns: Statistics data frame with stats fields
    """
    return get_stats(metrics, class_ids, None,save_csv=False)


def get_stats(metrics: MetricCollection, class_names: List[str], output: Path, save_csv: bool = True):
    """
    Get and save per class statistics
    Args:
        metrics: torchmetrics collection, has to have a `StatScores` metric
        class_names: List of class names
        output_dir: Output directory path
        save_csv: Whether to save as a csv file

    Returns: Statistics data frame
    """
    stats = metrics['StatScores']
    tp, fp, tn, fn = stats.tp.cpu().numpy(), stats.fp.cpu().numpy(), stats.tn.cpu().numpy(), stats.fn.cpu().numpy(),

    stats_data = {'TP': tp,
                  'FP': fp,
                  'TN': tn,
                  'FN': fn,
                  'Precision': tp / (tp + fp),
                  'Recall': tp / (tp + fn),
                  'F1': 2*tp / (2*tp + fp + fn),
                  'Total samples': tp + fn
                  }
    stats = pd.DataFrame(data=stats_data, index=class_names).fillna(0)  # fill NaNs with 0 in case tp + fp = 0

    if save_csv:
        # csv = Path(output_dir)/'eval_stats.csv'
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
