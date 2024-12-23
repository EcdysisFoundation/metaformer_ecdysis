import argparse
import logging
from pathlib import Path

from .data import BugBoxData
from .split import split_from_df, generate_split_class_report
from .utils import drop_identical_images

from . import LOGGING_LEVEL, INFO

SEED = 42
LOGGING_LEVEL = INFO

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)



def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Data generation pipeline for BugBox images')
    parser.add_argument('dataset', type=str, help='Name of the generated dataset')
    parser.add_argument('--bugbox-mnt', type=str, default='/pool1/srv/bugbox3/bugbox3/media/',
                        help='Path to BugBox images mounted directory')
    parser.add_argument('--train-size', type=float, default=0.6, help='Relative size of the train split')
    parser.add_argument('--drop-duplicates', action='store_true', help='Drop duplicate images from the dataset')
    parser.add_argument('--hard-copy', action='store_true', help='Copy images instead of symlinking them')
    parser.add_argument('--minimum-images', type=int, default=20, help='Do not create a class unless it has at least '
                                                                       'this number of images')

    return parser.parse_args()

def main():
    """ Generate a dataset from BugBox images """
    args = get_args()

    assert 0 < args.train_size <= 1, 'Train size must be between 0 and 1'

    bugbox_mnt = Path(args.bugbox_mnt)
    dataset_dir = Path(f'datasets/{args.dataset}')
    dataset_dir.mkdir(exist_ok=True, parents=True)

    db = BugBoxData()
    images = db.get_reviewed_images_df()
    images['image'] = images['image'].apply(lambda x: str(bugbox_mnt / x))

    if args.drop_duplicates:
        images = drop_identical_images(images)

    splits = split_from_df(images, args.train_size, dataset_dir, not args.hard_copy,
                   seed=SEED, min_images=args.minimum_images)


    meta_file = dataset_dir/'metadata.csv'
    images.to_csv(meta_file, index=False)

    morphospecies_map = db.get_morphospecies_df()
    morphospecies_map.to_csv(dataset_dir / 'morphospecies_map.csv')
    morphospecies_map.to_csv('deploy/morphospecies_map.csv') # here too for deployment .mar

    report_count_df = generate_split_class_report(splits, morphospecies_map)
    report_count_df.to_csv(dataset_dir + '/dataset_report.csv', index=False)

# This gets executed when running `python -m dataset_generation`
if __name__ == '__main__':
    main()
