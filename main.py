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
from src.exception import CustomException
from src.logger import logging

if __name__ == "__main__":
    training_pipeline_config = TrainingPipelineConfigEntity()

    data_ingestion_config = DataIngestionConfigEntity(
        training_pipeline_config=training_pipeline_config
    )

    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    data_validation_config = DataValidationConfigEntity(
        training_pipeline_config=training_pipeline_config
    )

    validation = DataValidation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_config=data_validation_config,
    )
    data_validation_artifact = validation.initiate_data_validation()

    data_transformation_config = DataTransformationConfigEntity(
        training_pipeline_config=training_pipeline_config,
    )

    transformation = DataTransformation(
        data_validation_artifact=data_validation_artifact,
        data_transformation_config=data_transformation_config,
    )
    data_transformation_artifact = transformation.initiate_data_transformation()

    model_trainer_config = ModelTrainerConfigEntity(
        training_pipeline_config=training_pipeline_config
    )
    model_trainer = ModelTrainer(
        model_trainer_config=model_trainer_config,
        data_transformation_artifact=data_transformation_artifact,
    )
    model_trainer_artifact = model_trainer.initiate_model_trainer()
