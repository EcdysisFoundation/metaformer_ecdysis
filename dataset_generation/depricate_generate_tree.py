import argparse
import hashlib
import logging
from pathlib import Path
from shutil import copy

from halo import Halo

import pandas as pd
from PIL import Image


from . import LOGGING_LEVEL, INFO

TAXON_LEVELS = levels = ['order', 'family', 'genus']
SEED = 42

LOGGING_LEVEL = INFO

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

def get_cli_arguments() -> argparse.Namespace:
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(description='Generate directory tree for classified images from BugBox')
    parser.add_argument('data_path', type=str, help='Path to the input data root directory')
    parser.add_argument('csv_path', type=str, help='Path to the csv file of insect data')
    parser.add_argument('output_path', type=str, help='Output directory path')
    parser.add_argument('--min-classification-level', type=str,
                        help='Minimum taxonomic min_classification_level needed to process an image',
                        default='family', choices=('family', 'genus'))
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    return parser.parse_args()


def generate_tree(data: pd.DataFrame, data_root: Path, output_dir: Path,
                  min_classification_level: str = None, symlink: bool = True) -> pd.DataFrame:
    """
    Download the classified data to output directory. Creates symbolic links by default for speed and to reduce disk
    usage.
    Args:
        data: DataFrame of BugBox image data
        data_root: Path to the images root directory (paths in `df` are relative to this path)
        output_dir: Path to output directory
        min_classification_level: Minimum taxon level allowed to process an image. Only relevant when the classification
        method is 'GBIF'
        symlink: Whether to use symbolic links or do hard copies

    Returns: Input DataFrame appended with the md5 column
    """
    if not data_root.is_dir():
        raise ValueError('Input directory not found')

    data = data.copy()
    data = data.dropna(subset=['morphos_name'])


    # process rows
    with Halo(text=f'Working on {len(data)} images', spinner='dots') as spinner:
        data['md5'] = data.apply(process_sample, args=(data_root, output_dir, symlink), axis=1)
        spinner.succeed('Done')

    return data


def process_sample(row: pd.Series, data_root: Path, output_dir: Path,
                   symlink: bool = True) -> str:
    """
    Retrieve and classify and individual sample. This function is ment to be used with the `apply` method of pandas
    DataFrame objects. Output image names have the format <specimen_id>_<index>.ext
    Args:
        row: Row of the data frame which must contain image; order, family and genus or morphospecie columns
        data_root: Path to the images root directory (paths in `df` are relative to this path)
        output_dir: Path to output directory
        symlink: Whether to use symbolic links or do hard copies

    Returns: MD5 hash of the image
    """
    dst_dir = output_dir/row['morphos_name']

    # Make directory if it does not already exist
    dst_dir.mkdir(parents=True, exist_ok=True)

    src = data_root / row['image']
    if not src.is_file():
        logger.warning(f'{src} file not found')
        return ''

    try:
        md5_id = hashlib.md5(Image.open(src).tobytes())
    except OSError:
        # logger.warning(f'Fail to open {src}')
        return ''

    # Using hashes as filenames implicitly avoids duplicates with the same classification, keeping the first entry
    output_image_name = f'{md5_id.hexdigest()}'

    dst = (dst_dir/output_image_name).with_suffix(src.suffix.lower())
    if dst.is_file():
        logger.debug(f'File {dst} already exists, skipping...')
        return output_image_name

    # Copy image
    logger.debug(f'Copying image {src} to {dst}')
    if symlink:
        dst.symlink_to(src)
    else:
        copy(src, dst)

    return output_image_name


if __name__ == '__main__':

    args = get_cli_arguments()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    df = pd.read_csv(args.csv_path)

    input_images = Path(args.data_path)
    output = Path(args.output_path)

    generate_tree(df, input_images, output)
