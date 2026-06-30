import pandas as pd

MORPHOS_ID = 'morphos_id'
MORPHOS_NAME = 'morphos_name'
DATASET_DIR = 'datasets'
MORPHOSPECIES_MAP = 'morphospecies_map.csv'
DF_COLS = [MORPHOS_NAME, MORPHOS_ID, 'specimen_id', 'image', 'gbif_canonical_name']


def get_reviewed_images_df():
    specimen_df = pd.read_csv('dataset_generation/training_selections.csv')
    return pd.DataFrame(specimen_df)[DF_COLS]


def get_supplemental_images_df():
    """
    Supplemental images to use if not enough in get_reviewed_images_df
    """
    extra_df = pd.read_csv('dataset_generation/gbif_training_selections.csv')
    return pd.DataFrame(extra_df)[DF_COLS]


def get_dataset(minimum_images):
    images_df = get_reviewed_images_df()
    images_supplemental = get_supplemental_images_df()

    # determine the maximum images per class, and which get supplemental images to get closer to that maximum
    df_counts = images_df[MORPHOS_NAME].value_counts()
    max_imgs_perclass = df_counts.max()

    if max_imgs_perclass < minimum_images:
        print(f"Exiting: The highest category count ({max_imgs_perclass}) is less than the required minimum of {minimum_images}.")
        return

    all_categories = set(images_df[MORPHOS_NAME].unique()).union(set(images_supplemental[MORPHOS_NAME].unique()))
    final_chunks = []

    for category in all_categories:
        existing_rows = images_df[images_df[MORPHOS_NAME] == category]
        current_count = len(existing_rows)

        # Calculate how many more we need to reach our dynamic maximum
        needed = max_imgs_perclass - current_count
        if needed > 0:
            # Filter images_supplemental for this specific category
            available_pool = images_supplemental[images_supplemental[MORPHOS_NAME] == category]
            # Take up to the 'needed' amount (handles cases where there aren't enough records)
            sampled_rows = available_pool.head(needed)
            # Merge the original rows and the top-up rows together
            combined_category_df = pd.concat([existing_rows, sampled_rows], ignore_index=True)
        else:
            # If images_df already has the maximum (or somehow more), cap it
            combined_category_df = existing_rows.head(max_imgs_perclass)

        if len(combined_category_df) >= minimum_images:
            final_chunks.append(combined_category_df)
        else:
            # By doing nothing, categories with less than 20 total images are dropped
            print(f"Dropped category '{category}' because total count ({len(combined_category_df)}) was below {minimum_images}.")

    # Combine everything together
    if final_chunks:
        final_df = pd.concat(final_chunks, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=images_df.columns)

    return final_df


def set_morphospecies_map_index(morphospecies_map):
    morphospecies_map[MORPHOS_ID] = morphospecies_map[MORPHOS_ID].astype('str')
    morphospecies_map.reset_index(drop=True)
    morphospecies_map = morphospecies_map.set_index(MORPHOS_ID)
    return morphospecies_map


def get_morphospecies_map(images_df):
    morphospecies_map = images_df[[MORPHOS_ID, MORPHOS_NAME]].drop_duplicates()
    return set_morphospecies_map_index(morphospecies_map)
