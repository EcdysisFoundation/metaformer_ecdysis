import logging
from typing import List

import pandas as pd
from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine import URL
from sshtunnel import SSHTunnelForwarder

from . import connection, queries

logging.basicConfig()
logger = logging.getLogger(__name__)


class BugBoxDB:
    """
    BugBox database object. Creates an SSH tunnel to ecdysis01 and connects to the Postgres database.

    Args:
        local_port: Local port to bind remote's database port
    """
    def __init__(self, local_port: int = 5433):
        self.server = SSHTunnelForwarder(
            (connection.server, 22),
            ssh_username=connection.ssh_username,
            ssh_password=connection.ssh_password,
            #remote_bind_address=(connection.pg_host, connection.pg_port),
            remote_bind_address=('localhost', connection.pg_port),
            local_bind_address=('localhost', local_port)
            )
        self.url = URL.create(
            drivername='postgresql',
            username=connection.pg_user,
            password=connection.pg_password,
            host='localhost',
            port=local_port,
            database=connection.db
        )
        self.engine = create_engine(self.url)
        self.connection = None
        self.meta = MetaData()

    def connect(self):
        """Starts the SSH tunnel and connects to de DB"""
        self.server.start()
        self.connection = self.engine.connect()
        logger.info('Connected to BugBox database')

    def disconnect(self):
        """Close connection and SSH tunnel"""
        self.connection.close()
        self.engine.dispose()
        self.server.stop()
        logger.info('Connection closed')

    def query_df(self, sql_query: str):
        """
        Make an SQL query to the DB

        Args:
            sql_query: SQL query as a string

        Returns: Panda's DataFrame with the results of the query
        """
        if self.connection is None:
            logger.info('Starting database connection')
            self.connect()

        data = pd.read_sql(sql_query, con=self.connection)

        return data

    def get_tables(self, tables: List[str] = None):
        """
        Get the descriptions of all or a subset of tables from the database

        Args:
            tables: List of table names, if `None` all tables are returned

        Returns: Metadata object
        """
        self.meta.reflect(only=tables)

        return self.meta.tables

    def get_reviewed_images_df(self, columns: List[str] = None, lookback: str = None) -> pd.DataFrame:
        """
        Get table of reviewed images

        Args:
            columns: Subset of columns
            lookback: Interval to query for new images; could be day, week, month, year. If None, it does not perform
            filtering by date

        Returns: Pandas DataFrame
        """
        if lookback is not None:
            last_uploaded_query = queries.images.substitute(lookback=lookback)
        else:
            last_uploaded_query = queries.images.substitute(lookback='century')

        reviewed_images = self.query_df(queries.images_with_taxon.substitute(image_query=last_uploaded_query))

        if columns is not None:
            reviewed_images = reviewed_images[columns]

        return reviewed_images

    def get_reference_images_df(self) -> pd.DataFrame:
        """
        Get table of reference images

        Returns: Pandas DataFrame
        """
        reference_images = self.query_df(queries.reference_images)

        return reference_images

    def get_images_df(self, columns: List[str] = None, lookback: str = 'week') -> pd.DataFrame:
        """
        Get table of reviewed images

        Args:
            columns: Subset of columns
            lookback: Interval to query for new images; could be day, week, month, year. If None, it does not perform
            filtering by date

        Returns: Pandas DataFrame
        """
        if lookback is not None:
            last_uploaded_query = queries.images.substitute(lookback=lookback)
        else:
            last_uploaded_query = queries.images.substitute(lookback='century')

        images = self.query_df(last_uploaded_query)

        if columns is not None:
            images = images[columns]

        return images

    def get_taxa_df(self, columns: List[str] = None):
        """
        Get table of taxon ids

        Args:
            columns: Subset of columns

        Returns: Pandas DataFrame
        """

        taxa = self.query_df(queries.taxa)

        if columns is not None:
            taxa = taxa[columns]

        return taxa

    def get_morphospecies_df(self, columns: List[str] = None):
        """
        Get table of morphospecies ids

        Args:
            columns: Subset of columns

        Returns: Pandas DataFrame
        """

        morphospecies = self.query_df(queries.morphospecies)

        if columns is not None:
            morphospecies = morphospecies[columns]

        return morphospecies


if __name__ == '__main__':
    db = BugBoxDB()
    db.connect()

    images = db.get_reviewed_images_df(columns=['image', 'morphospecie_id'], lookback='week')

    print(images.head())

    db.disconnect()
