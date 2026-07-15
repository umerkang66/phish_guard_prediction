import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

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

            print(f"Model Trainer Artifact: {model_trainer_artifact}")

            logging.info("Training pipeline completed successfully.")
        except Exception as e:
            raise CustomException(e, sys)
