import os, sys
from typing import List

import pymongo
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from src.exception import CustomException
from src.logger import logging

# configuration for data ingestion
from src.entity.config_entity import DataIngestionConfigEntity
from src.entity.artifact_entity import DataIngestionArtifactEntity

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfigEntity):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def export_collection_as_dataframe(self) -> List[pd.DataFrame]:
        try:
            logging.info("Getting the data from MongoDB")
            client = pymongo.MongoClient(DATABASE_URL)
            db = client[self.data_ingestion_config.database_name]
            collection = db[self.data_ingestion_config.collection_name]
            data = list(collection.find())
            logging.info("Data has been fetched from MongoDB")

            logging.info("Converting the data to pandas DataFrame")
            df = pd.DataFrame(data)
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"])
            df.replace({"na": np.nan}, inplace=True)

            logging.info("Data has been converted to pandas DataFrame")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def export_data_to_features_store(self, df: pd.DataFrame) -> None:
        try:
            logging.info("Exporting the data to features store")

            os.makedirs(
                os.path.dirname(self.data_ingestion_config.feature_store_file_path),
                exist_ok=True,
            )

            df.to_csv(
                self.data_ingestion_config.feature_store_file_path,
                index=False,
                header=True,
            )
            logging.info("Data has been exported to features store")

        except Exception as e:
            raise CustomException(e, sys)

    def split_data_as_train_test(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        try:
            logging.info("Splitting the data into train and test sets")
            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_ratio,
                random_state=42,
            )

            logging.info("Data has been split into train and test sets")

            logging.info("Exporting the train and test sets to features store")
            os.makedirs(
                os.path.dirname(self.data_ingestion_config.train_file_path),
                exist_ok=True,
            )
            train_set.to_csv(
                self.data_ingestion_config.train_file_path, index=False, header=True
            )
            os.makedirs(
                os.path.dirname(self.data_ingestion_config.test_file_path),
                exist_ok=True,
            )
            test_set.to_csv(
                self.data_ingestion_config.test_file_path, index=False, header=True
            )
            logging.info("Train and test sets have been exported to features store")

            return train_set, test_set

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifactEntity:
        try:
            logging.info("Starting data ingestion")

            df = self.export_collection_as_dataframe()
            self.export_data_to_features_store(df)
            self.split_data_as_train_test(df)

            dataingestion_artifact = DataIngestionArtifactEntity(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path,
            )

            logging.info("Data ingestion completed")

            return dataingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)
