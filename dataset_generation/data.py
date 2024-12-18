from typing import List

import pandas as pd


class BugBoxData:

    def get_df(self: str):
        """
        Make a dataframe from a csv
        Returns: Panda's DataFrame with the results of the query
        """
        # replace path with dataset_generation/training_selections.csv for final edit
        specimen_df = pd.read_csv('dataset_generation/training_selections_tests/training_selections_1.csv')
        return pd.DataFrame(specimen_df)

    def get_reviewed_images_df(self):
        reviewed_images = self.get_df()
        return reviewed_images[['morphos_name', 'morphos_id', 'specimen_id', 'image', 'specimen_count']]

    def get_taxon_map_df(self):
        taxon_map = self.get_df()
        return taxon_map[['morphos_id', 'morphos_name','specimen_id']]

    def get_morphospecies_df(self):
        morphospecies = self.get_df()
        morphospecies = morphospecies[['morphos_id', 'morphos_name']].drop_duplicates()
        morphospecies.morphos_id = morphospecies.morphos_id.astype('str')
        morphospecies = morphospecies.set_index('morphos_id')
        return morphospecies


