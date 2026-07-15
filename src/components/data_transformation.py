import sys, os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from src.constants.training_pipeline_constants import TARGET_COLUMN
from src.constants.training_pipeline_constants import DATA_TRANSFORMATION_IMPUTER_PARAMS

from src.entity.artifact_entity import (
    DataTransformationArtifactEntity,
    DataValidationArtifactEntity,
)

from src.entity.config_entity import DataTransformationConfigEntity

from src.exception import CustomException
from src.logger import logging
from src.utils import save_numpy_array_data, save_object, load_object


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifactEntity,
        data_transformation_config: DataTransformationConfigEntity,
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            # Create a KNN imputer with specified parameters
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)

            # Create a pipeline with the imputer
            pipeline = Pipeline(steps=[("imputer", imputer)])

            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifactEntity:
        try:
            logging.info("Starting data transformation process.")
            # Load the train and test data
            train_df = self.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = self.read_data(
                self.data_validation_artifact.valid_test_file_path,
            )

            # Separate features and target variable
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]
            y_train = y_train.replace(-1, 0)
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]
            y_test = y_test.replace(-1, 0)

            # Get the data transformer object
            data_transformer = self.get_data_transformer_object()

            # Fit the transformer on the training data and transform both train and test data
            X_train_imputed = data_transformer.fit_transform(X_train)
            X_test_imputed = data_transformer.transform(X_test)

            train_arr = np.c_[X_train_imputed, y_train.to_numpy()]
            test_arr = np.c_[X_test_imputed, y_test.to_numpy()]

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path, test_arr
            )
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                data_transformer,
            )

            logging.info("Ended data transformation process.")

            # preparing artifacts
            return DataTransformationArtifactEntity(
                preprocessing_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
