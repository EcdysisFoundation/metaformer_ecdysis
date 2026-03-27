import logging
from hashlib import md5
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

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
    cache_path = Path('.cache/hashes.csv')

    # 1. Load hashes
    if cache_path.is_file():
        logger.info('Loading hashes from cache')
        cache_df = pd.read_csv(cache_path).drop_duplicates(subset=['image'])
        cache_map = cache_df.set_index('image')['hash']
    else:
        cache_map = pd.Series(dtype=str)

    # 2. Map existing hashes to our current data
    data['hash'] = data['image'].map(cache_map)

    # 3. Only process images that are still null
    null_mask = data['hash'].isnull()
    num_to_check = null_mask.sum()

    if num_to_check > 0:
        logger.info(f'Calculating hashes for {num_to_check} new images')
        data.loc[null_mask, 'hash'] = data.loc[null_mask, 'image'].apply(get_md5_hash)

        # 4. Update the persistent cache with ONLY unique new entries
        # We combine the old cache with new results and drop duplicates
        new_hashes = data[['image', 'hash']].dropna()
        updated_cache = pd.concat([cache_df if cache_path.is_file() else None, new_hashes])
        updated_cache.drop_duplicates(subset=['image']).to_csv(cache_path, index=False)

    # 5. Drop duplicates from the current session based on the hash
    num_starting = len(data)
    output = data.drop_duplicates(subset=['hash'])

    logger.info(f'Dropped {num_starting - len(output)} duplicated images')
    return output


def is_image_corrupted(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # Verify the image structure
        return False  # Image is not corrupted
    except (IOError, SyntaxError) as e:
        logger.error(e)
        return True  # Image is corrupted
