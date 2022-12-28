"""
Split and copy insect image files from ML_Dataset directory in ecdysis01 server to its assigned split. The output
directory structure follows Imagenet format. Example:

    datasets/insectagen/
    ├── meta.yaml
    ├── test
    │    ├── Eribolus
    │    ├── Liohippelates
    │    └── Oscinella
    ├── train
    │    ├── Eribolus
    │    ├── Liohippelates
    │    └── Oscinella
    └── val
        ├── Eribolus
        ├── Liohippelates
        └── Oscinella

"""
import argparse
import logging
from pathlib import Path
from shutil import copy, SameFileError

import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEED = 42


def parse_options():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(description='Generate splits from insecta data.')
    parser.add_argument('input_directory', type=str, help=f'Path to input directory. Images must be inside '
                                                          'subdirectories named as the class they' f' belong to.')
    parser.add_argument('output_directory', type=str, help='Output directory')
    parser.add_argument('--train-size', type=float, help='Relative size of the trainig set. Val/Test sizes are computed'
                                                         ' as (1-train_size)/2.', default=0.8)
    parser.add_argument('--ignore-augmentations', action='store_true', help='Ignore augmented images. Use if the classes'
                                                                            'have regular and augmented images divided'
                                                                            'in folders')
    parser.add_argument('--debug', action='store_true', help='Enable logging of debug messages.')
    parser.add_argument('--seed', type=int, default=SEED, help='Random state')
    parser.add_argument('--no-copy', action='store_false', help='Do not copy images to output directory')
    parser.add_argument('--yaml-file', type=str, default='meta.yaml', help='Name of output yaml file')

    return parser.parse_args()


def get_image_paths(input: Path, ignore_augmentations: bool = False) -> dict:
    """
    Get images paths from input root directory
    Args:
        input: Path to root directory
        ignore_augmentations: Ignore directories of the form root/class/class_*/ containing augmented images

    Returns: Dictionary of classes to its list of image paths
    """
    images_by_class = {}
    for cls_dir in input.iterdir():
        if cls_dir.is_dir():  # Avoid files on the root directory
            name = cls_dir.name
            non_augmented_images = cls_dir/name if ignore_augmentations else cls_dir
            images = list(non_augmented_images.rglob('*.jpg'))
            LOGGER.info(f'{len(images)} images for family {name}')
            images_by_class[name] = list(map(str, images))  # Need the paths as strings for writing yaml

    return images_by_class


def save_class_images(splits: dict, class_name: str, output: Path = Path('datasets/') / 'data'):
    """
    Save images of a class divided in splits
    Args:
        splits: Dictionary of lists of image paths per split
        class_name: Name of the class
        output: Path to output directory
    """
    def copy_img(src: Path, dst: Path):
        LOGGER.debug(f'Copying {src} to {dst}')
        try:
            copy(src, dst)
        except SameFileError:
            LOGGER.warning(f'File {dst} already present, skipping')

    for split_name, split_img in splits[class_name].items():
        parent = output / split_name / class_name
        parent.mkdir(parents=True, exist_ok=True)

        LOGGER.debug(f'Writing images to {parent}')
        for img in tqdm(split_img,
                        desc=f'Copying {len(split_img)} {split_name} images of {class_name.replace("_", " ")} class'):
            src = Path(img)
            dst = parent / src.name
            copy_img(src, dst)


def random_split(images_by_class: dict, train_size: float, output: Path = Path('datasets')/'data',
                 copy: bool = True, save_yaml: bool = True, seed: int = SEED, **kwargs):
    """
    Split images of a dataset in train/val/test. The splitting preserves the distribution of samples per class in each
    group (stratification).
    Args:
        images_by_class: Dictionary of image paths for every class
        train_size: Proportion of images reserved for train. Val/Test sizes are computed as (1 - train_size)/2
        output: Path to output directory
        copy: Copy images
        save_yaml: Create yaml metadata file
        seed: Random state
        **kwargs: For yaml file name pass `yaml_name` as keyword argument
    """
    assert (copy or save_yaml), 'At least one of `copy` or `save_yaml` must be set to `True`'
    assert 0 < train_size <= 1, 'Train size must be between 0 and 1'

    splits = {}
    for name, images in images_by_class.items():
        train, test_val = train_test_split(images, train_size=train_size, random_state=seed)
        val, test = train_test_split(test_val, test_size=0.5, random_state=seed)
        splits[name] = {'train': train, 'val': val, 'test': test}

        if copy:
            save_class_images(splits, name, output)

        if save_yaml:
            yaml_name = kwargs.get('yaml_name', 'meta.yaml')
            save_yaml_file({'seed': seed, 'splits': splits}, output, yaml_name)


def save_yaml_file(data: dict, output_dir: Path, name: str):
    with (output_dir/name).open('w') as f:
        yaml.dump(data, f)


if __name__ == '__main__':

    args = parse_options()

    # Logging config
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    LOGGER = logging.getLogger(__name__)

    input_root = Path(args.input_directory)
    output = Path(args.output_directory)

    # Get images
    images = get_image_paths(input_root, args.ignore_augmentations)

    # Split and save
    random_split(images, args.train_size, output, args.no_copy, seed=args.seed, yaml_name=args.yaml_file)
