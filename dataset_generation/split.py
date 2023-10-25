"""
Split and make_directory_tree insect image files from a directory  to its assigned split. The output directory structure follows
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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset_generation.utils import save_yaml_file
from . import SEED, TAXON_LEVELS, LOGGING_LEVEL
from typing import List, Tuple, Dict


logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def parse_options():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(description='Generate splits from insecta data.')

    parser.add_argument('input_directory', type=str, help=f'Path to input directory. Images must be inside '
                                                          'subdirectories named as the class they belong to.')
    parser.add_argument('output_directory', type=str, help='Output directory')

    parser.add_argument('--train-size', type=float, help='Relative size of the trainig set. Val/Test sizes are computed'
                                                         ' as (1-train_size)/2.', default=0.7)
    parser.add_argument('--level', type=str, help='Maximum taxon rank to be used as labels', choices=('family', 'genus')
                        , default='genus')
    parser.add_argument('--min-images', type=int, help='Minimum number of image of a class for it to be considered '
                                                       'well represented', default=5)
    parser.add_argument('--debug', action='store_true', help='Enable logging of debug messages.')
    parser.add_argument('--seed', type=int, default=SEED, help='Random state')
    parser.add_argument('--no-make_directory_tree', action='store_false', help='Do not make_directory_tree images to output directory')
    parser.add_argument('--follow-symlinks', action='store_true', help='Set it to do copies even if the input is a'
                                                                       'symbolic link')
    parser.add_argument('--yaml-file', type=str, default='splits.yaml', help='Name of output yaml file')

    parser.add_argument('--add-reference-images', action='store_true', help='Add reference images to train split')
    parser.add_argument('--reference-image-path', type=str, help='Path to the reference images root directory',
                        default='/home/ecdysis/ref_images_classified')

    return parser.parse_args()


def get_image_paths(input: Path, depth: int) -> dict:
    """
    Get images paths from input root directory, returning a multi-level dictionary with class and subclasses as keys.
    Args:
        input: Path to root directory
        depth: Levels down the directory tree to use as labels, samples that don't have at least this many levels are
    ignored

    Returns: Dictionary of lists of image paths per class
    """
    images_by_class = {}
    images = input.rglob('*.jpg')
    for image in images:
        levels = image.relative_to(input).parts
        if len(levels) >= depth:  # +1 for the filename itself
            class_name = ' '.join([levels[i] for i in range(depth)])
            if class_name in images_by_class:
                images_by_class[class_name].append(str(image))
            else:
                images_by_class[class_name] = [str(image)]

    logger.info(f'{len(images_by_class)} classes found')

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


def remove_class(images: dict, class_name: str, target_class_name: str = 'incertae sedis') -> dict:
    """
    Moves all samples of the class to class 'incertae sedis'
    Args:
        images: Dictionary of lists of image paths per class
        class_name: Name of the class
        target_class_name: Name of the target class

    Returns: Updated dictionary

    """
    class_images = images.pop(class_name)
    if target_class_name in images:
        # append to existing list
        images[target_class_name] += class_images
    else:
        # class_images is already a list, so we don't need to create one
        images[target_class_name] = class_images

    return images


def filter_underrepresented(images: dict, threshold: int, target_class_name: str = 'incertae sedis') -> dict:
    """
    Move all underrepresented classes to class 'incertae sedis'
    Args:
        images: Dictionary of lists of image paths per class (class_name/id: list of image paths)
        threshold: Minimum number of samples for the class to be considered as well represented

    Returns: Updated dictionary, list of underrepresented classes

    """
    underrepresented_list = []
    for class_name in list(images.keys()):
        if is_class_underrepresented(images, class_name, threshold):
            images = remove_class(images, class_name, target_class_name)
            underrepresented_list.append(class_name)
    return images,underrepresented_list


def save_class_images(splits: dict, class_name: str, output: Path = Path('datasets') / 'data',
                      use_symlinks: bool = True):
    """
    Save images of a class divided in splits
    Args:
        splits: Dictionary of lists of image paths per split
        class_name: Name of the class
        output: Path to output directory
    """
    def copy_img(src: Path, dst: Path):
        logger.debug(f'Copying {src} to {dst}')
        try:
            copy(src, dst, follow_symlinks=True)
        except SameFileError:
            logger.warning(f'File {dst} already present, skipping')

    for split_name, split_img in splits[class_name].items():
        if len(split_img) == 0:
            continue
        parent = output / split_name / class_name
        parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f'Writing images to {parent}')
        for img in tqdm(split_img,
                        desc=f'Copying {len(split_img)} {split_name} images of {class_name.replace("_", " ")} class'):
            src = Path(img)
            dst = parent / src.name
            if not dst.is_file():
                if use_symlinks:
                    dst.symlink_to(src)
                else:
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
        logger.debug(f'Reference images found for class {class_name}')
        splits[class_name]['train'] += list(images)
    else:
        logger.warning(f'No reference images found for class {class_name}')

    return splits


def random_split(images_by_class: dict, train_size: float, add_refimages: bool, reference_image_dir: Path = Path('.'),
                 output: Path = Path('datasets')/'data', make_directory_tree: bool = True, use_symlinks: bool = True,
                 save_yaml: bool = True, seed: int = SEED, **kwargs):
    """
    Split images of a dataset in train/val/test. The splitting preserves the distribution of samples per class in each
    group (stratification).
    Args:
        images_by_class: Dictionary of image paths for every class
        train_size: Proportion of images reserved for train. Val/Test sizes are computed as (1 - train_size)/2
        add_refimages: Set to True to add reference images to train split
        output: Path to output directory
        make_directory_tree: Copy images
        save_yaml: Create yaml metadata file
        seed: Random state
        **kwargs: For yaml file name pass `yaml_name` as keyword argument
    """
    assert (make_directory_tree or save_yaml), 'At least one of `make_directory_tree` or `save_yaml` must be set to `True`'
    assert 0 < train_size <= 1, 'Train size must be between 0 and 1'

    splits = {}
    for name, images in images_by_class.items():

        if len(images) > 1:
            train, test_val = train_test_split(images, train_size=train_size, random_state=seed)

            # If there is not at least one image for val and test append all to train
            if len(test_val) >= 2:
                val, test = train_test_split(test_val, train_size=0.5, random_state=seed)
            else:
                train += test_val
                val = []
                test = []
        else:
            train = images
            val = []
            test = []

        splits[name] = {'train': train, 'val': val, 'test': test}

        if add_refimages:
            splits = add_reference_images(splits, name, Path(reference_image_dir))

        if make_directory_tree:
            save_class_images(splits, name, output, use_symlinks)

    if save_yaml:
        yaml_name = kwargs.get('yaml_name', 'splits.yaml')
        save_yaml_file({'seed': seed, 'splits': splits}, output, yaml_name)


def split_from_directory(root: Path, output: Path, train_size: float, depth: int, add_reference_images: bool = True,
                         remove_underrepresented_classes: bool = False, make_directory_tree: bool = False, **kwargs):

    """
    Read images from directory tree in Imagenet format and make train/val/test splits
    Args:
        root: Path to input root directory
        output: Path to output directory
        train_size: Relative size of train subset. Validation and test relative size is computed as (1 - train_size) / 2
        depth: Number of levels down the input directory tree to use for classification. Images above this level are
            ignored
        add_reference_images: Whether to add reference images to the training split. If True, the keyword argument
            `reference_images` with the correct path must be added
        remove_underrepresented_classes: Whether to generate a class if only enough samples are available. Pass the
            `min_images` keyword argument with the minimum threshold (defaults to 20)
        make_directory_tree: If True, make new directory tree respecting splits in the output directory. Otherwise, only
            a yaml file is saved (used mostly for testing and debugging)
    """

    images = get_image_paths(root, depth)

    if remove_underrepresented_classes:
        images = filter_underrepresented(images, kwargs.get('min_images', 20))

    if add_reference_images:
        reference_images = Path(kwargs.get('reference_images', '.'))
        random_split(images, train_size, True, reference_images, output=output, make_directory_tree=make_directory_tree, seed=SEED)
    else:
        random_split(images, train_size, False, output=output, make_directory_tree=make_directory_tree, seed=SEED)

def get_count_per_class_split(splits:Dict[str, Dict[str, List[str]]]) -> pd.DataFrame:
    """
    Get the number of images per class in each split
    splits has the following format (as in the splits.yaml file)
    {
     '99': {
            'test': [  '/path/to/test_image1_for_class_99.jpg',... ],
            'train': [  '/path/to/train_image1_for_class_99.jpg',...  ],
            'val': [  '/path/to/val_image1_for_class_99.jpg', ...]
        },
        ...
    }
    Args:
        splits: Dictionary of lists of image paths per split, the key is the class name, the value is a dict of split, list of image path of that split and class
    Returns:
        Dataframe with the number of images per class in each split, columns are split names (train,test,val), rows are class ids
    """
    counts = []
    for class_id, split in splits.items():
        # id, train, test, val
        counts.append({"id":class_id,**{split_name: len(image_paths) for split_name, image_paths in split.items()}})
    return pd.DataFrame(counts)

def generate_split_class_report(splits, taxon_map:pd.DataFrame):
    """
    Return the dataset sample count report
    Merges the split count with the morpho-species name

    Args:
        splits: A dictionary of splits (Dict[str, List[str]])
        taxon_map: A dataframe including the morpho-species name and id (pd.DataFrame)
    Returns:
        pd.DataFrame: A dataframe including  morpho-species name and the count of samples in each split (test, train, val)
    """
    counts_df = get_count_per_class_split(splits)
    counts_df["id"] = counts_df["id"].astype(int)
    counts_df = taxon_map[['id', 'name']].merge(counts_df, left_on='id', right_on='id', how='right')
    counts_df["total_samples"] = counts_df["train"] + counts_df["val"] + counts_df["test"]
    return counts_df.sort_values(by="name")

def split_from_df(df: pd.DataFrame, train_size: float, output: Path, use_symlinks: bool, classification_method: str,
                  refimages: pd.DataFrame = None, save_yaml: bool = True, seed: int = SEED, **kwargs):
    """
    Split images of a dataset in train/val/test. The splitting preserves the distribution of samples per class in each
    group (stratification).
    Args:
        df: Input DataFrame, the output of `db.get_reviewed_images`
        train_size: Proportion of images reserved for train. Val/Test sizes are computed as (1 - train_size)/2
        output: Path to output directory
        save_yaml: Create yaml splits file
        classification_method: Either 'GBIF' or 'morphospecie'
        seed: Random state
        **kwargs: For yaml file name pass `yaml_name` as keyword argument
    """
    if not 0 < train_size <= 1:
        raise ValueError('Train size must be between 0 and 1')

    df = df.copy()

    if classification_method == 'GBIF':
        # Generate class names and add reference images
        rank = kwargs.get('rank', 'genus')
        taxon_ranks = TAXON_LEVELS[:TAXON_LEVELS.index(rank) + 1]
        if refimages is not None:
            # Only use refimages of rank present in BugBox images
            df = pd.concat([df, refimages[refimages[rank].isin(df[rank])]], axis=0)
        df.replace('', np.nan, inplace=True)  # Handle empty strings
        df.dropna(subset=taxon_ranks, inplace=True)
        df['class_name'] = df[taxon_ranks].agg(' '.join, axis=1)
        name_column = 'class_name'
        underrepresented_name = 0
    elif classification_method == 'morphospecie':
        df.replace('', np.nan, inplace=True)  # Handle empty strings
        df.dropna(subset=['morphospecie_id'], inplace=True)
        df['class_name'] = df['morphospecie_id']
        name_column = "morphospecies"
        underrepresented_name = 3458
    else:
        raise ValueError(f'\"{classification_method}\" is not a valid classification method')
    # When using morphospecies class name here is actually the id of the class, converted to str, not the name
    images = dict(df.groupby('class_name')['image'].apply(list))  # Convert to dict to use filter_underrepresented
    # Moves the underrepresented classes to incertae sedis
    images, underrepresented_classes = filter_underrepresented(images, kwargs.get('min_images', 20), underrepresented_name)
    
    # Save which classes are replaced by incertae sedis and how many sample in total are in each class
    underrepresented_df = df[df['class_name'].isin(underrepresented_classes)][[name_column]].value_counts().reset_index(name="total_samples").sort_values(["total_samples",name_column],ascending=[False,True])
    underrepresented_df.to_csv(output / 'underrepresented_classes.csv',index=False)
    
    splits = {}


    for class_name, image_list in images.items():
        class_name = str(class_name)

        train, test_val = train_test_split(image_list, train_size=train_size, random_state=seed)
        val, test = train_test_split(test_val, train_size=0.5, random_state=seed)

        splits[class_name] = {'train': train, 'val': val, 'test': test}

        save_class_images(splits, class_name, output, use_symlinks)

    if save_yaml:
        yaml_name = kwargs.get('yaml_name', 'splits.yaml')
        save_yaml_file({'seed': seed, 'splits': splits}, output, yaml_name)
    return splits


if __name__ == '__main__':

    args = parse_options()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    input_root = Path(args.input_directory)
    output = Path(args.output_directory)

    n_levels = TAXON_LEVELS.index(args.level)

    # Get images
    images = get_image_paths(input_root, n_levels)
    images = filter_underrepresented(images, args.min_images)

    logger.info(f'{len(images) - 1} non underrepresented classes after filtering (threshold={args.min_images})')
    logger.info(f'{len(images["incertae sedis"])} images were unassigned during filtering')

    # Split and save
    random_split(images, args.train_size, args.add_reference_images, args.reference_image_path, output, args.no_copy,
                 args.follow_symlinks, seed=args.seed, yaml_name=args.yaml_file)
