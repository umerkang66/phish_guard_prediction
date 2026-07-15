import os
import sys
import json
import certifi
from dotenv import load_dotenv

import pandas as pd
import pymongo

from src.exception import CustomException
from src.logger import logging

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

ca = certifi.where()


class NetworkDataExtract:
    def __init__(self, database, collection_name):
        self.database = database
        self.collection_name = collection_name
        self.mongo_client = None
        if not self.mongo_client:
            self.mongo_client = pymongo.MongoClient(DATABASE_URL, tlsCAFile=ca)

    def csv_to_json_converter(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True)

            records = list(json.loads(df.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e, sys)

    def insert_data_to_mongodb(self, data):
        try:
            db = self.mongo_client[self.database]
            collection = db[self.collection_name]
            collection.insert_many(data)
            logging.info(
                f"Data inserted into MongoDB collection: {self.collection_name}"
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        database = "network_data"
        collection_name = "network_metrics"
        file_path = "Network_Data/phishingData.csv"

        extractor = NetworkDataExtract(database, collection_name)
        data = extractor.csv_to_json_converter(file_path)
        extractor.insert_data_to_mongodb(data)

    except Exception as e:
        logging.error(f"Error occurred: {e}")
