import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_obj
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    model_obj_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('splitting the data into x_train, y_train, x_test, y_test')
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # models dictionary
            models = {
                'knn': KNeighborsRegressor(),
                'decision_tree': DecisionTreeRegressor(),
                'random_forest': RandomForestRegressor(),
                'ada_boost': AdaBoostRegressor(),
                'gradient_boosting': GradientBoostingRegressor(),
                'svr': SVR(),
                'linear_regression': LinearRegression(),
                'catboost': CatBoostRegressor(verbose=False),
                'xgboost': XGBRegressor(),
            }

            models_report: dict = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
            )

            best_model = max(models_report, key=models_report.get)
            best_model_score = models_report[best_model]

            if best_model_score < 0.6:
                raise CustomException('Best model score is less than 0.6')

            logging.info(f'Best model found: {best_model}')

            save_obj(
                obj=models[best_model],
                file_path=self.model_trainer_config.model_obj_file_path,
            )

            return best_model_score
        except Exception as e:
            raise CustomException(e, sys)
