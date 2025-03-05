import csv
import logging
from pathlib import Path
from typing import List

import pandas as pd
from torchmetrics import Accuracy, Precision, Recall, F1Score, StatScores, MetricCollection
from yacs.config import CfgNode

from datetime import datetime

from dataset_generation.data import BugBoxData, MORPHOS_ID


def get_model_metrics(config: CfgNode):
    """
    Get metric collection for model training and evaluation
    Args:
        config: Model's configuration file

    Returns: torchmetrics collection
    """
    metrics = [Accuracy(task="multiclass", num_classes=config.MODEL.NUM_CLASSES, average='micro'),
               Precision(task="multiclass", num_classes=config.MODEL.NUM_CLASSES, average='macro'),
               Recall(task="multiclass", num_classes=config.MODEL.NUM_CLASSES, average='macro'),
               F1Score(task="multiclass", num_classes=config.MODEL.NUM_CLASSES, average='macro')]

    if config.EVAL_MODE:
        metrics.append(StatScores(task="multiclass", num_classes=config.MODEL.NUM_CLASSES, average='macro'))

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
    if 'MulticlassStatScores' in metrics.keys():
        stats = metrics['MulticlassStatScores']
        tp, fp, tn, fn = stats.tp.cpu().numpy(), stats.fp.cpu().numpy(), stats.tn.cpu().numpy(), stats.fn.cpu().numpy()

        return {'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn,
                'Precision': tp / (tp + fp),
                'Recall': tp / (tp + fn),
                'F1': 2*tp / (2*tp + fp + fn),
                total_column_name: tp + fn
            }
    else:
        # return zeros if MulticlassStatScores not entered, and print keys in case not as expected
        print(metrics.keys())
        return {'TP': 0,
                'FP': 0,
                'TN': 0,
                'FN': 0,
                'Precision': 0,
                'Recall': 0,
                'F1': 0,
                total_column_name: 0
        }

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
    stats_data = _get_stats_from_metrics(metrics,'Total_samples')
    stats = pd.DataFrame(data=stats_data, index=class_names).fillna(0)  # fill NaNs with 0 in case tp + fp = 0

    if save_csv:
        db = BugBoxData()
        morphospecies_df = db.get_morphospecies_df()
        stats['model_name'] = version
        stats = stats.merge(morphospecies_df, how='left', left_index=True, right_index=True)
        stats_csv = 'stats.csv'
        stats.to_csv(output / stats_csv)
        dataset_report_ = 'dataset_report_'
        dataset_report_path = output/f'dataset_report.csv'
        dataset_report_df = pd.read_csv(dataset_report_path) if dataset_report_path.exists() else None
        dataset_report_df.morphos_id = dataset_report_df.morphos_id.astype('str')
        print('dataset_report_df: ' + str(len(dataset_report_df)))
        logging.info(f"Length of dataset_report: {len(dataset_report_df)}, stats: {len(stats)}")
        dataset_report_df = dataset_report_df.add_prefix(dataset_report_)
        dataset_report_morphos_id = dataset_report_ + MORPHOS_ID
        dataset_report_df = dataset_report_df.set_index(dataset_report_morphos_id)
        stats = stats.merge(dataset_report_df, how='left', left_index=True, right_index=True)
        v = dataset_report_ + stats_csv
        stats.to_csv(output / v)
    return stats


def log_metrics(logger: logging.Logger, metrics: MetricCollection, aggregation: str):
    """
    Simple function to log classification metrics at batch or epoch level
    Args:
        logger:
        metrics:
        aggregation:

    Returns:

    """
    metrics_string = ' | '.join(f'{m} = {v}' for m, v in metrics.items() if m not in ['MulticlassStatScores'])
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
