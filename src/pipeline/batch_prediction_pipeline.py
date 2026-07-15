import os
import sys
import pandas as pd
import yaml
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.constants.training_pipeline_constants import TARGET_COLUMN, SCHEMA_FILE_PATH

class BatchPredictionPipeline:
    def __init__(self):
        self.preprocessing_path = os.path.join("final_model", "preprocessing.pkl")
        self.model_path = os.path.join("final_model", "model.pkl")

    def model_files_exist(self) -> bool:
        """Checks if the preprocessing object and trained model files exist."""
        return os.path.exists(self.preprocessing_path) and os.path.exists(self.model_path)

    def get_schema_features(self) -> list:
        try:
            with open(SCHEMA_FILE_PATH, "r") as f:
                schema = yaml.safe_load(f)
            columns = []
            for col_dict in schema.get("columns", []):
                for col_name in col_dict.keys():
                    if col_name != TARGET_COLUMN:
                        columns.append(col_name)
            return columns
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Accepts a pandas DataFrame containing features (and optionally the target).
        Performs batch prediction and returns a copy of the input DataFrame
        with predicted target values appended as new columns.
        """
        try:
            if not self.model_files_exist():
                raise Exception("Model or Preprocessing objects do not exist. Please train the model first.")

            # Read features from schema
            feature_cols = self.get_schema_features()

            # Validate that all required features exist in the input dataframe
            missing_cols = [col for col in feature_cols if col not in dataframe.columns]
            if missing_cols:
                raise ValueError(f"Uploaded CSV is missing the following required columns: {missing_cols}")

            # Keep only the features in the correct order for preprocessing
            df_features = dataframe[feature_cols].copy()

            logging.info("Loading preprocessor and model for batch prediction.")
            preprocessor = load_object(self.preprocessing_path)
            model = load_object(self.model_path)

            logging.info("Transforming batch dataframe using preprocessor.")
            transformed_data = preprocessor.transform(df_features)

            logging.info("Generating batch predictions.")
            predictions = model.predict(transformed_data)

            # Create a copy of dataframe to append predictions
            result_df = dataframe.copy()
            
            # Map predictions: 0 represents Phishing (-1), and 1 represents Legitimate (1)
            result_df['Predicted_Value'] = [1 if p == 1 else -1 for p in predictions]
            result_df['Predicted_Label'] = ["Legitimate" if p == 1 else "Phishing" for p in predictions]

            return result_df

        except Exception as e:
            raise CustomException(e, sys)
