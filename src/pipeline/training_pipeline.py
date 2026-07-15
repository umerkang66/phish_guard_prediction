import sys
from datetime import datetime

from src.cloud.s3_syncer import S3Sync
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.constants.training_pipeline_constants import TRAINING_BUCKET_NAME

from src.entity.config_entity import (
    DataIngestionConfigEntity,
    ModelTrainerConfigEntity,
    TrainingPipelineConfigEntity,
    DataValidationConfigEntity,
    DataTransformationConfigEntity,
)

from src.entity.artifact_entity import (
    DataIngestionArtifactEntity,
    DataValidationArtifactEntity,
    DataTransformationArtifactEntity,
    ModelTrainerArtifactEntity,
)


class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfigEntity()
            self.s3_sync = S3Sync()
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifactEntity:
        try:
            self.data_ingestion_config = DataIngestionConfigEntity(
                training_pipeline_config=self.training_pipeline_config,
            )

            data_ingestion = DataIngestion(self.data_ingestion_config)

            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifactEntity
    ) -> DataValidationArtifactEntity:
        try:
            self.data_validation_config = DataValidationConfigEntity(
                training_pipeline_config=self.training_pipeline_config,
            )

            data_validation = DataValidation(
                data_validation_config=self.data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )

            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifactEntity
    ) -> DataTransformationArtifactEntity:
        try:
            self.data_transformation_config = DataTransformationConfigEntity(
                training_pipeline_config=self.training_pipeline_config,
            )

            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )

            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifactEntity
    ) -> ModelTrainerArtifactEntity:
        try:
            self.model_trainer_config = ModelTrainerConfigEntity(
                training_pipeline_config=self.training_pipeline_config,
            )

            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )

            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise CustomException(e, sys)

    def sync_artifact_dir_to_s3(self):
        try:
            logging.info("Syncing artifact directory to S3 started.")
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifacts{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.artifact_dir,
                aws_bucket_url=aws_bucket_url,
            )
            logging.info("Syncing artifact directory to S3 completed successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def sync_saved_model_dir_to_s3(self):
        try:
            logging.info("Syncing saved model directory to S3 started.")
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            self.s3_sync.sync_folder_to_s3(
                folder="final_model",  # Assuming saved models are stored in a directory named 'final_model'
                aws_bucket_url=aws_bucket_url,
            )
            logging.info("Syncing saved model directory to S3 completed successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Training pipeline Started.")

            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )

            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            print(f"Model Trainer Artifact: {model_trainer_artifact}")

            logging.info("Training pipeline completed successfully.")
        except Exception as e:
            raise CustomException(e, sys)
