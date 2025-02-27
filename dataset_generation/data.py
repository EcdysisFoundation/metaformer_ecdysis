from typing import List

import pandas as pd

MORPHOS_ID = 'morphos_id'
MORPHOS_NAME = 'morphos_name'

class BugBoxData:

    def get_df(self: str):
        """
        Make a dataframe from a csv
        Returns: Panda's DataFrame with the results of the query
        """
        specimen_df = pd.read_csv('dataset_generation/training_selections.csv')
        return pd.DataFrame(specimen_df)

    def get_reviewed_images_df(self):
        reviewed_images = self.get_df()
        return reviewed_images[[MORPHOS_NAME, MORPHOS_ID, 'specimen_id', 'image', 'specimen_count']]

    def get_morphospecies_df(self):
        morphospecies = self.get_df()
        # warning, code expects this order of ['morphos_id', 'morphos_name'].
        morphospecies = morphospecies[[MORPHOS_ID, MORPHOS_NAME]].drop_duplicates()
        morphospecies.morphos_id = morphospecies.morphos_id.astype('str')
        morphospecies.reset_index(drop=True)
        morphospecies = morphospecies.set_index(MORPHOS_ID)
        return morphospecies


