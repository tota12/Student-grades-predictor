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
                'linear_regression': LinearRegression(fit_intercept=True),
                'catboost': CatBoostRegressor(verbose=False),
                'xgboost': XGBRegressor(),
            }

            params = {
                'knn': {'n_neighbors': [2, 4, 6, 8]},
                'decision_tree': {
                    'max_depth': [2, 4, 6, 8, 10],
                },
                'random_forest': {'n_estimators': [8, 16, 32, 64, 128, 256]},
                'ada_boost': {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                'gradient_boosting': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                'svr': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                'linear_regression': {},
                'catboost': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
                'xgboost': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
            }
            models_report: dict = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params,
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
