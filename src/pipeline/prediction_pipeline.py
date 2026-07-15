import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, NetworkModel

class PredictionPipeline:
    def __init__(self):
        self.preprocessing_path = os.path.join("final_model", "preprocessing.pkl")
        self.model_path = os.path.join("final_model", "model.pkl")

    def model_files_exist(self) -> bool:
        """Checks if the preprocessing object and trained model files exist."""
        return os.path.exists(self.preprocessing_path) and os.path.exists(self.model_path)

    def predict(self, dataframe: pd.DataFrame) -> int:
        """
        Accepts a pandas DataFrame containing all features and makes a prediction.
        Returns 1 for Legitimate and -1/0 for Phishing.
        """
        try:
            if not self.model_files_exist():
                raise Exception("Model or Preprocessing objects do not exist. Please train the model first.")

            logging.info("Loading preprocessor and model for prediction.")
            preprocessor = load_object(self.preprocessing_path)
            model = load_object(self.model_path)

            logging.info("Generating predictions using NetworkModel.")
            network_model = NetworkModel(preprocessor=preprocessor, model=model)
            prediction = network_model.predict(dataframe)
            return int(prediction[0])
        except Exception as e:
            raise CustomException(e, sys)
