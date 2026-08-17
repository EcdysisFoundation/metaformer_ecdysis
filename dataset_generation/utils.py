import logging
from hashlib import md5
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from . import LOGGING_LEVEL
from .data import MD5_HASH, IMAGE_FIELD

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def get_pixel_md5(image_field: str) -> str:
    # Calling image_field.open('rb') guarantees Django fetches
    # the file stream from local storage OR Amazon S3.
    with image_field.open('rb') as f:
        with Image.open(f) as img:
            img_standardized = img.convert('RGB')
            pixel_bytes = img_standardized.tobytes()
            return md5(pixel_bytes).hexdigest()


def save_yaml_file(data: dict, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/name).open('w') as f:
        yaml.dump(data, f)


def drop_identical_images(data: pd.DataFrame):
    """
    Uses the md5 hash of the image to remove duplicates
    Args:
        data: DataFrame containing the image paths in the IMAGE_FIELD column

    Returns: DataFrame with duplicates removed and a new column 'hash' containing the md5 hash of the image

    """
    cache_path = Path('.cache/hashes.csv')

    # 1. Load hashes
    if cache_path.is_file():
        logger.info('Loading hashes from cache')
        cache_df = pd.read_csv(cache_path).drop_duplicates(subset=[IMAGE_FIELD])
        cache_map = cache_df.set_index(IMAGE_FIELD)[MD5_HASH]
    else:
        cache_map = pd.Series(dtype=str)

    # 2. Map existing hashes to our current data
    data[MD5_HASH] = data[IMAGE_FIELD].map(cache_map)

    # 3. Only process images that are still null
    null_mask = data[MD5_HASH].isnull()
    num_to_check = null_mask.sum()

    if num_to_check > 0:
        logger.info(f'Calculating hashes for {num_to_check} new images')
        # 1. Identify missing values (handles NaN, None, and empty strings)
        missing_mask = data[MD5_HASH].isna() | (data[MD5_HASH].astype(str).str.strip() == '')
        # 2. Compute the MD5 hash ONLY for rows where md5_hash is missing
        data.loc[missing_mask, MD5_HASH] = data.loc[missing_mask, IMAGE_FIELD].apply(get_pixel_md5)
        # 4. Update the persistent cache with ONLY unique new entries
        # We combine the old cache with new results and drop duplicates
        new_hashes = data[[IMAGE_FIELD, MD5_HASH]].dropna()
        updated_cache = pd.concat([cache_df if cache_path.is_file() else None, new_hashes])
        updated_cache.drop_duplicates(subset=[IMAGE_FIELD]).to_csv(cache_path, index=False)

    # 5. Drop duplicates from the current session based on the hash
    num_starting = len(data)
    output = data.drop_duplicates(subset=[MD5_HASH])

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
