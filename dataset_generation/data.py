from typing import List

import pandas as pd


class BugBoxData:

    def get_df(self: str):
        """
        Make a dataframe from a csv
        Returns: Panda's DataFrame with the results of the query
        """

        specimen_df = pd.read_csv('dataset_generation/training_selections_tests/training_selections_1.csv')
        return pd.DataFrame(specimen_df)

    def get_reviewed_images_df(self, columns: List[str] = None) -> pd.DataFrame:
        """
        Get table of reviewed images
        Args:
            columns: Subset of columns
        Returns: Pandas DataFrame
        """

        reviewed_images = self.get_df()
        reviewed_images = reviewed_images[['morphos_name', 'morphos_id', 'specimen_id', 'image', 'specimen_count']]

        if columns is not None:
            reviewed_images = reviewed_images[columns]

        return reviewed_images


    def get_morphospecies_df(self, columns: List[str] = None):
        """
        Get table of morphospecies ids
        Args:
            columns: Subset of columns

        Returns: Pandas DataFrame
        """

        morphospecies = self.get_df()

        if columns is not None:
            morphospecies = morphospecies[columns]

        return morphospecies
