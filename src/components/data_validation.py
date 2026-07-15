import os, sys
from scipy.stats import ks_2samp
import pandas as pd

from src.entity.artifact_entity import (
    DataIngestionArtifactEntity,  # this for the input to validation class
    DataValidationArtifactEntity,
)
from src.constants.training_pipeline_constants import SCHEMA_FILE_PATH
from src.entity.config_entity import DataValidationConfigEntity
from src.utils import read_yaml_file, write_yaml_file

from src.exception import CustomException
from src.logger import logging


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifactEntity,
        data_validation_config: DataValidationConfigEntity,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {dataframe.columns}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise CustomException(e, sys)

    def detect_data_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05,
    ) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update(
                    {
                        column: {
                            "p_value": float(is_same_dist.pvalue),
                            "drift_status": is_found,
                        }
                    }
                )
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifactEntity:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # read the data from train and test file path
            train_df = self.read_data(train_file_path)
            test_df = self.read_data(test_file_path)

            # validate the number of columns
            train_status = self.validate_number_of_columns(train_df)
            test_status = self.validate_number_of_columns(test_df)

            if not train_status or not test_status:
                raise Exception(
                    "Data Validation failed. Number of columns are not as per schema."
                )

            # let's check data drift using Kolmogorov-Smirnov test
            drift_status = self.detect_data_drift(base_df=train_df, current_df=test_df)

            if not drift_status:
                raise Exception(
                    "Data Validation failed. Data drift is detected between train and test data."
                )

            dir_path = os.path.dirname(
                self.data_validation_config.valid_train_file_path,
            )
            os.makedirs(dir_path, exist_ok=True)
            train_df.to_csv(
                path_or_buf=self.data_validation_config.valid_train_file_path,
                index=False,
                header=True,
            )

            dir_path = os.path.dirname(
                self.data_validation_config.valid_test_file_path,
            )
            os.makedirs(dir_path, exist_ok=True)
            test_df.to_csv(
                path_or_buf=self.data_validation_config.valid_test_file_path,
                index=False,
                header=True,
            )

            return DataValidationArtifactEntity(
                validation_status=train_status and test_status and drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
