import argparse
import logging
import os
from pathlib import Path

from .data import (
    get_dataset, get_morphospecies_map,
    DATASET_DIR, MORPHOS_NAME,
    MORPHOSPECIES_MAP
)
from .split import split_from_df, generate_split_class_report
from .utils import drop_identical_images, is_image_corrupted

from . import LOGGING_LEVEL, SEED


logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Data generation pipeline')
    parser.add_argument('dataset', type=str, help='Name of the generated dataset')
    parser.add_argument('--img-mnt', type=str,
                        help='Path to images directory')
    parser.add_argument('--train-size', type=float, default=0.6, help='Relative size of the train split')
    parser.add_argument('--check-corrupted', action='store_true', help='Check for corrupted images')
    parser.add_argument('--hard-copy', action='store_true', help='Copy images instead of symlinking them')
    parser.add_argument('--minimum-images', type=int, default=20, help='Do not create a class unless it has at least '
                                                                       'this number of images')

    return parser.parse_args()


def main():
    """ Generate a dataset from images """
    args = get_args()

    assert 0 < args.train_size <= 1, 'Train size must be between 0 and 1'

    img_mnt = Path(args.img_mnt)
    dataset_dir = Path(f'{DATASET_DIR}/{args.dataset}')
    dataset_dir.mkdir(exist_ok=True, parents=True)

    images = get_dataset(args.minimum_images)

    images['image'] = images['image'].apply(lambda x: str(img_mnt / x))

    # check if files exist
    print('Checking for missing images ...')
    images['exists'] = images['image'].astype(str).map(os.path.exists)
    missing_images = images[images['exists'] == False]
    if len(missing_images):
        v = len(missing_images)
        if v >= 20:
            v = 20
        print('some images are missing. Up to the first 20 are...')
        print(missing_images.iloc[0:v])
        print('saving to file in dataset_dir ....')
        missing_images.to_csv(dataset_dir / 'missing_images.csv')
        print('exiting...........')
        return

    if args.check_corrupted:
        # check for corrupted files
        print('checking for corrupted images ....')
        images['corrupted'] = images['image'].astype(str).map(is_image_corrupted)
        corrupted_images = images[images['corrupted']]
        if len(corrupted_images):
            corrupted_images.to_csv(dataset_dir / 'corrupted_images.csv')
            print('Some images are corrupted, see report file, exiting ....')
            return

    images = drop_identical_images(images)

    # re-check minimum images
    image_counts = images[MORPHOS_NAME].value_counts()
    valid_categories = image_counts[image_counts >= args.minimum_images].index
    images = images[images[MORPHOS_NAME].isin(valid_categories)]

    splits = split_from_df(images, args.train_size, dataset_dir, not args.hard_copy,
                           seed=SEED, min_images=args.minimum_images)

    meta_file = dataset_dir/'metadata.csv'
    images.to_csv(meta_file, index=False)

    morphospecies_map = get_morphospecies_map(images)
    morphospecies_map.to_csv(dataset_dir / MORPHOSPECIES_MAP)

    report_count_df = generate_split_class_report(splits, morphospecies_map)
    report_count_df.to_csv(dataset_dir / 'dataset_report.csv', index=False)


if __name__ == '__main__':
    main()
