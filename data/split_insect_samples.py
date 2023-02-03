"""
Split and copy insect image files from a directory  to its assigned split. The output directory structure follows
Imagenet format. Example:

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

logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_options():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(description='Generate splits from insecta data.')

    parser.add_argument('input_directory', type=str, help=f'Path to input directory. Images must be inside '
                                                          'subdirectories named as the class they belong to.')
    parser.add_argument('output_directory', type=str, help='Output directory')

    parser.add_argument('--train-size', type=float, help='Relative size of the trainig set. Val/Test sizes are computed'
                                                         ' as (1-train_size)/2.', default=0.7)
    parser.add_argument('--levels', type=int, help='Number of levels down the input root directory to be used as labels'
                        , default=1)
    parser.add_argument('--min-images', type=int, help='Minimum number of image of a class for it to be considered '
                                                       'well represented', default=5)
    parser.add_argument('--debug', action='store_true', help='Enable logging of debug messages.')
    parser.add_argument('--seed', type=int, default=SEED, help='Random state')
    parser.add_argument('--no-copy', action='store_false', help='Do not copy images to output directory')
    parser.add_argument('--follow-symlinks', action='store_true', help='Set it to do copies even if the input is a'
                                                                       'symbolic link')
    parser.add_argument('--yaml-file', type=str, default='splits.yaml', help='Name of output yaml file')

    parser.add_argument('--add-reference-images', action='store_true', help='Add reference images to train split')
    parser.add_argument('--reference-image-path', type=str, help='Path to the reference images root directory',
                        default='/home/ecdysis/ref_images_classified')

    return parser.parse_args()


def get_image_paths(input: Path, n_levels: int) -> dict:
    """
    Get images paths from input root directory, returning a multi-level dictionary with class and subclasses as keys.
    Args:
        input: Path to root directory
        n_levels: Levels down the directory tree to use as labels, samples that don't have at least this many levels are
    ignored

    Returns: Dictionary of lists of image paths per class
    """
    images_by_class = {}
    images = input.rglob('*.jpg')
    for image in images:
        taxon_levels = image.relative_to(input).parts
        if len(taxon_levels) >= n_levels + 1:
            class_name = ' '.join([taxon_levels[i] for i in range(n_levels)])
            if class_name in images_by_class:
                images_by_class[class_name].append(str(image))
            else:
                images_by_class[class_name] = [str(image)]

    LOGGER.info(f'{len(images_by_class)} classes found')

    return images_by_class


def is_class_underrepresented(images: dict, class_name: str, threshold: int) -> bool:
    """
    Assess if the class in the dictionary has fewer samples than the threshold
    Args:
        images: Dictionary of lists of image paths per class
        class_name: Name of the class
        threshold: Minimum number of samples for the class to be considered as well represented

    Returns: True if the class is underrepresented

    """
    number_of_images = len(images[class_name])

    return number_of_images < threshold


def declassify_class(images: dict, class_name: str) -> dict:
    """
    Moves all samples of the class to class 'other'
    Args:
        images: Dictionary of lists of image paths per class
        class_name: Name of the class

    Returns: Updated dictionary

    """
    class_images = images.pop(class_name)

    if 'other' in images:
        images['other'] += class_images
    else:
        images['other'] = class_images

    return images


def filter_uderrepresented(images: dict, threshold: int) -> dict:
    """
    Move all underrepresented classes to class 'other'
    Args:
        images: Dictionary of lists of image paths per class
        threshold: Minimum number of samples for the class to be considered as well represented

    Returns: Updated dictionary

    """
    for class_name in list(images.keys()):
        if is_class_underrepresented(images, class_name, threshold):
            images = declassify_class(images, class_name)

    return images


def save_class_images(splits: dict, class_name: str, output: Path = Path('datasets') / 'data',
                      follow_symlinks: bool = True):
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
            copy(src, dst, follow_symlinks=follow_symlinks)
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


def add_reference_images(splits: dict, class_name: str, reference_images_dir: Path) -> dict:
    """
    Add reference images of a class to the train split
    Args:
        splits: Dictionary of lists of image paths per split
        class_name: Name of the class
        reference_images_dir: Path to reference images root directory

    Returns: Updated splits dictionary
    """
    image_directory = reference_images_dir.joinpath(*class_name.split())

    if image_directory.is_dir():
        images = map(str, image_directory.rglob('*.jpg'))
        LOGGER.debug(f'Reference images found for class {class_name}')
        splits[class_name]['train'] += list(images)

    return splits


def random_split(images_by_class: dict, train_size: float, add_refimages: bool, reference_image_dir: str = '.',
                 output: Path = Path('datasets')/'data', copy: bool = True, follow_symlinks: bool = True,
                 save_yaml: bool = True, seed: int = SEED, **kwargs):
    """
    Split images of a dataset in train/val/test. The splitting preserves the distribution of samples per class in each
    group (stratification).
    Args:
        images_by_class: Dictionary of image paths for every class
        train_size: Proportion of images reserved for train. Val/Test sizes are computed as (1 - train_size)/2
        add_reference_images: Set to True to add reference images to train split
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
        LOGGER.debug(f'{name}:\n{images}')
        train, test_val = train_test_split(images, train_size=train_size, random_state=seed)
        val, test = train_test_split(test_val, train_size=0.5, random_state=seed)
        splits[name] = {'train': train, 'val': val, 'test': test}

        if add_refimages:
            splits = add_reference_images(splits, name, Path(reference_image_dir))

        if copy:
            save_class_images(splits, name, output, follow_symlinks)

    if save_yaml:
        yaml_name = kwargs.get('yaml_name', 'meta.yaml')
        save_yaml_file({'seed': seed, 'splits': splits}, output, yaml_name)


def save_yaml_file(data: dict, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/name).open('w') as f:
        yaml.dump(data, f)


if __name__ == '__main__':

    args = parse_options()

    if args.debug:
        LOGGER.setLevel(logging.DEBUG)

    input_root = Path(args.input_directory)
    output = Path(args.output_directory)

    # Get images
    images = get_image_paths(input_root, args.levels)
    images = filter_uderrepresented(images, args.min_images)

    LOGGER.info(f'{len(images)-1} non underrepresented classes after filtering (threshold={args.min_images})')
    LOGGER.info(f'{len(images["other"])} images were unassigned during filtering')

    # Split and save
    random_split(images, args.train_size, args.add_reference_images, args.reference_image_path, output, args.no_copy,
                 args.follow_symlinks, seed=args.seed, yaml_name=args.yaml_file)
