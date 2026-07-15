import os, sys

from src.exception import CustomException
from src.logger import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import mlflow

from src.entity.artifact_entity import (
    ClassificationMetricArtifactEntity,
    DataTransformationArtifactEntity,
    ModelTrainerArtifactEntity,
)
from src.entity.config_entity import ModelTrainerConfigEntity


from src.utils import (
    evaluate_models,
    save_object,
    load_object,
    load_numpy_array_data,
    get_classification_score,
    NetworkModel,
)

import dagshub

dagshub.init(repo_owner="ugulzar4512", repo_name="phish_guard_prediction", mlflow=True)


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfigEntity,
        data_transformation_artifact: DataTransformationArtifactEntity,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e

    def track_mlflow(self, best_model, metrics: ClassificationMetricArtifactEntity):
        try:
            with mlflow.start_run():
                accuracy = metrics.accuracy
                f1_score = metrics.f1_score
                precision = metrics.precision_score
                recall = metrics.recall_score

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

                if isinstance(best_model, XGBClassifier):
                    mlflow.xgboost.log_model(best_model, "model")
                else:
                    mlflow.sklearn.log_model(best_model, "model")

        except Exception as e:
            raise CustomException(e, sys)

    def train_model(
        self, X_train, y_train, X_test, y_test
    ) -> ModelTrainerArtifactEntity:
        try:
            models = {
                "RandomForest": RandomForestClassifier(verbose=1, n_jobs=-1),
                "XGBoost": XGBClassifier(),
                "CatBoost": CatBoostClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
                "GradientBoosting": GradientBoostingClassifier(
                    verbose=1,
                ),
                "AdaBoost": AdaBoostClassifier(),
                "LogisticRegression": LogisticRegression(verbose=1, n_jobs=-1),
            }
            params = {
                "RandomForest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                },
                "XGBoost": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                },
                "CatBoost": {
                    "iterations": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 8],
                },
                "DecisionTree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                },
                "GradientBoosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample": [0.6, 0.8, 1.0],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                },
                "LogisticRegression": {
                    "C": [0.01, 0.1, 1, 10],
                    "penalty": ["l2"],
                    "solver": ["lbfgs", "liblinear"],
                },
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            best_model_score = max(sorted([v[0] for v in model_report.values()]))
            best_model_name = list(model_report.keys())[
                [v[0] for v in model_report.values()].index(best_model_score)
            ]

            best_model = models[best_model_name]
            best_model.set_params(**model_report[best_model_name][1])

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            classification_train_metric = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred,
            )

            classification_test_metric = get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred,
            )

            self.track_mlflow(
                best_model=best_model,
                metrics=classification_train_metric,
            )
            self.track_mlflow(
                best_model=best_model,
                metrics=classification_test_metric,
            )

            preprocessor = load_object(
                file_path=self.data_transformation_artifact.preprocessing_object_file_path
            )

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path)
            )

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)

            save_object(
                self.model_trainer_config.trained_model_file_path, obj=network_model
            )

            final_model_dir = os.path.join(os.getcwd(), "final_model")
            os.makedirs(final_model_dir, exist_ok=True)

            # save the best model for prediction
            save_object(os.path.join(final_model_dir, "model.pkl"), best_model)

            return ModelTrainerArtifactEntity(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifactEntity:
        try:
            logging.info("Initiating model trainer component")
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model_trainer_artifact = self.train_model(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
            )

            logging.info("Ended model training")

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
