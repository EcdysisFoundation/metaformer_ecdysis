import argparse
import hashlib
import logging
from pathlib import Path
from shutil import copy
from typing import List

import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(format='%(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)


def get_cli_arguments() -> argparse.Namespace:
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(description='Generate directory tree for classified image data')

    parser.add_argument('data_path', type=str, help='Path to the input data root directory')
    parser.add_argument('csv_path', type=str, help='Path to the csv file of insect data')
    parser.add_argument('output_path', type=str, help='Output directory path')

    parser.add_argument('--top-hierarchy', type=str, help='Top hierarchy level for classification', default='order')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    return args


def generate_data_directories(data: pd.DataFrame, data_root: Path, output_dir: Path, level: str, symlink: bool = True):
    """
    Download the classified data to output directory. Creates symbolic links by default for speed and to reduce disk
    usage.
    Args:
        data: DataFrame of BugBox image data
        data_root: Path to the images root directory (paths in `df` are relative to this path)
        output_dir: Path to output directory
        level: Lowest taxon level to use for classification
        symlink: Whether to use symbolic links or do hard copies

    """
    if not data_root.is_dir():
        raise ValueError('Input directory not found')

    LOGGER.info(f'{len(data)} samples to be retrieved')
    data = data.dropna(subset=['order', 'family'])
    data = data.fillna('')  # To avoid problems when the genus is empty

    # process rows
    data.apply(process_sample, args=(data_root, output_dir, level, symlink), axis=1)


def process_sample(row: pd.Series, data_root: Path, output_dir: Path, level: str, symlink: bool = True) -> None:
    """
    Retrieve and classify and individual sample. This function is ment to be used with the `apply` method of pandas
    DataFrame objects
    Args:
        row: Row of the data frame which must contain image, order, family and genus columns
        data_root: Path to the images root directory (paths in `df` are relative to this path)
        output_dir: Path to output directory
        level: Lowest taxon level to use for classification
        symlink: Whether to use symbolic links or do hard copies

    """
    levels = ['order', 'family', 'genus']
    level_idx = levels.index(level)
    dst_dir = output_dir.joinpath(*[row[l] for l in levels[level_idx:]])

    # Make directory if not already exists
    dst_dir.mkdir(parents=True, exist_ok=True)

    src = data_root / row['image'].lstrip('/')  # TODO remove preceding slash during csv generation
    if not src.is_file():
        LOGGER.warning(f'{src} file not found')
        return

    if 'uuid' in row:
        output_image_name = f'{row["uuid"]}_{row.name}.jpg'
    else:
        md5_id = hashlib.md5(Image.open(src).tobytes())
        output_image_name = f'{md5_id.hexdigest()}_{row.name}.jpg'

    dst = dst_dir/output_image_name
    if dst.is_file():
        LOGGER.warning(f'File {dst} already exists, skipping...')
        return

    # Copy image
    LOGGER.debug(f'Copying image {src} to {dst}')
    if symlink:
        dst.symlink_to(src)
    else:
        copy(src, dst)


if __name__ == '__main__':

    args = get_cli_arguments()

    if args.debug:
        LOGGER.setLevel(logging.DEBUG)

    df = pd.read_csv(args.csv_path)

    input_images = Path(args.data_path)
    output = Path(args.output_path)

    generate_data_directories(df, input_images, output, args.top_hierarchy)

    LOGGER.info('Done')
