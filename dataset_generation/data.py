import logging
from typing import List

import pandas as pd
# local imports
from . import LOGGING_LEVEL, INFO

TAXON_LEVELS = levels = ['order', 'family', 'genus']
SEED = 42
LOGGING_LEVEL = INFO

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


class BugBoxData:

    def get_df(self: str):
        """
        Make a dataframe from a csv


        Returns: Panda's DataFrame with the results of the query
        """

        specimen_df = pd.read_csv('data/specimen_data.csv')
        return specimen_df

    def get_reviewed_images_df(self, columns: List[str] = None) -> pd.DataFrame:
        """
        Get table of reviewed images

        Args:
            columns: Subset of columns
            lookback: Interval to query for new images; could be day, week, month, year. If None, it does not perform
            filtering by date

        Returns: Pandas DataFrame
        """

        reviewed_images = self.get_df()
        
        reviewed_images = reviewed_images[['morphos_name','morphos_id','specimen_id','uuid','image', 'specimen_count']]

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


if __name__ == '__main__':
    df = BugBoxData()

    images = df.get_reviewed_images_df(columns=[['image', 'morphos_id']], lookback='week')

    print(images.head())

