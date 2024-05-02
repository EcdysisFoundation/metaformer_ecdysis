import argparse
import logging
from pathlib import Path

from . import LOGGING_LEVEL, SEED, TAXON_LEVELS
from .db import BugBoxDB
from .split import split_from_df, generate_split_class_report
from .utils import drop_identical_images

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)



def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Data generation pipeline for BugBox images')

    parser.add_argument('classification_method', type=str, choices=('GBIF', 'morphospecie'),
                        help="Classify images using GBIF's ids or Ecdysis's morphospecies")
    parser.add_argument('--taxon-rank', type=str, default='genus', choices=('family', 'genus'),
                        help='Taxonomic rank to use for classification, only used for GBIF method')
    parser.add_argument('--bugbox-mnt', type=str, default='/pool1/srv/bugbox/',
                        help='Path to BugBox images mounted directory')
    parser.add_argument('--train-size', type=float, default=0.6, help='Relative size of the train split')
    parser.add_argument('--dataset-name', type=str, default='bugbox', help='Name of the generated dataset')
    parser.add_argument('--reference-image', type=str, default='/pool1/ref_images/gen2-19-vcm/',
                        help='Path to reference images root directory')
    parser.add_argument('--lookback-interval', type=str, default=None, choices=('day', 'week', 'month', 'year'),
                        help='Time interval to query for new images; could be day, week, month, year')
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
    refimage_dir = Path(f'{args.reference_image}')
    dataset_dir = Path(f'datasets/{args.dataset_name}')

    dataset_dir.mkdir(exist_ok=True, parents=True)

    db = BugBoxDB()

    images = db.get_reviewed_images_df()
    images['image'] = images['image'].apply(lambda x: str(bugbox_mnt / x))

    if args.drop_duplicates:
        images = drop_identical_images(images)

    if args.classification_method == 'GBIF':
        refimages = db.get_reference_images_df()
        refimages['image'] = refimages['image'].apply(lambda x: str(refimage_dir / x))
    else:
        refimages = None

    splits = split_from_df(images, args.train_size, dataset_dir, not args.hard_copy, args.classification_method,
                  rank=args.taxon_rank, seed=SEED, refimages=refimages, min_images=args.minimum_images)


    meta_file = dataset_dir/'metadata.csv'
    images.to_csv(meta_file, index=False)

    taxon_map = db.get_morphospecies_df(columns=['id', 'name', 'taxon_id'])
    taxon_map.to_csv('deploy/taxon_map.csv', index=False)

    report_count_df = generate_split_class_report(splits, taxon_map)
    report_count_df.to_csv(dataset_dir / 'dataset_report.csv', index=False)

    db.disconnect()


# This gets executed when running `python -m dataset_generation`
if __name__ == '__main__':
    main()