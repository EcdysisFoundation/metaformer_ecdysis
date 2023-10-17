import logging
from hashlib import md5
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from halo import Halo

from . import LOGGING_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def get_md5_hash(image_path: str):
    try:
        hash = md5(Image.open(image_path).tobytes())
    except (IOError, OSError):
        logger.warning(f'Failed to process file: {image_path}')
        return

    return hash.hexdigest()


def save_yaml_file(data: dict, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/name).open('w') as f:
        yaml.dump(data, f)


def drop_identical_images(data: pd.DataFrame):
    """
    Uses the md5 hash of the image to remove duplicates
    Args:
        data: DataFrame containing the image paths in the 'image' column

    Returns: DataFrame with duplicates removed and a new column 'hash' containing the md5 hash of the image

    """
    # Check if we have a cache of hashes
    if Path('.cache/hashes.csv').is_file():
        logger.info('Loading hashes from cache')
        hashes = pd.read_csv('.cache/hashes.csv')
        data = data.merge(hashes, on='image', how='left')
    else:
        data['hash'] = None

    num_starting_images = len(data)
    with Halo(text=f'Removing duplicates, {num_starting_images} images to check', spinner='dots') as spinner:
        null_hash_mask = data['hash'].isnull()
        data.loc[null_hash_mask, 'hash'] = data.loc[null_hash_mask, 'image'].apply(get_md5_hash)
        data[['image', 'hash']].to_csv('.cache/hashes.csv', index=False)
        output = data.drop_duplicates(subset=['hash'])
        spinner.succeed(f'---------Dropped {num_starting_images - len(output)} duplicated images---------')

    return output
