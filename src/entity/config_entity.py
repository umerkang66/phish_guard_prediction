from datetime import datetime
import os
from src.constants import training_pipeline_constants


class TrainingPipelineConfigEntity:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.pipeline_name = training_pipeline_constants.PIPELINE_NAME
        self.artifact_name = training_pipeline_constants.ARTIFIACT_DIR
        self.artifact_dir = os.path.join(
            training_pipeline_constants.ARTIFIACT_DIR, self.timestamp
        )


class DataIngestionConfigEntity:
    def __init__(self, training_pipeline_config: TrainingPipelineConfigEntity):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline_constants.DATA_INGESTION_DIR_NAME,
        )
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline_constants.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline_constants.FILENAME,
        )
        self.train_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline_constants.DATA_INGESTION_INGESTED_DIR,
            training_pipeline_constants.TRAIN_FILENAME,
        )
        self.test_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline_constants.DATA_INGESTION_INGESTED_DIR,
            training_pipeline_constants.TEST_FILENAME,
        )
        self.train_test_ratio = (
            training_pipeline_constants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        )
        self.collection_name = (
            training_pipeline_constants.DATA_INGESTION_COLLECTION_NAME
        )
        self.database_name = training_pipeline_constants.DATA_INGESTION_DATABASE_NAME


class DataValidationConfigEntity:
    def __init__(self, training_pipeline_config: TrainingPipelineConfigEntity):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline_constants.DATA_VALIDATION_DIR_NAME,
        )
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline_constants.DATA_VALIDATION_VALID_DIR,
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline_constants.DATA_VALIDATION_INVALID_DIR,
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir, training_pipeline_constants.TRAIN_FILENAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir, training_pipeline_constants.TEST_FILENAME
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir, training_pipeline_constants.TRAIN_FILENAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir, training_pipeline_constants.TEST_FILENAME
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline_constants.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline_constants.DATA_VALIDATION_DRIFT_REPORT_FILENAME,
        )


class DataTransformationConfigEntity:
    def __init__(self, training_pipeline_config: TrainingPipelineConfigEntity):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline_constants.DATA_TRANSFORMATION_DIR_NAME,
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline_constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline_constants.TRAIN_FILENAME.replace("csv", "npy"),
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline_constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline_constants.TEST_FILENAME.replace("csv", "npy"),
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline_constants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline_constants.PREPROCESSING_OBJECT_FILENAME,
        )


class ModelTrainerConfigEntity:
    def __init__(self, training_pipeline_config: TrainingPipelineConfigEntity):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline_constants.MODEL_TRAINER_DIR_NAME,
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline_constants.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline_constants.MODEL_TRAINER_TRAINED_MODEL_NAME,
        )
        self.expected_accuracy: float = (
            training_pipeline_constants.MODEL_TRAINER_EXPECTED_SCORE
        )
        self.overfitting_underfitting_threshold = (
            training_pipeline_constants.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
        )
