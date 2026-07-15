from dataclasses import dataclass


@dataclass
class DataIngestionArtifactEntity:
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifactEntity:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifactEntity:
    preprocessing_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassificationMetricArtifactEntity:
    accuracy: float
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifactEntity:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifactEntity
    test_metric_artifact: ClassificationMetricArtifactEntity
