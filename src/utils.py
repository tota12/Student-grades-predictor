import os
import sys
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import dill


def save_obj(obj, file_path):
    '''
    This method is responsible for saving the object to the file
    '''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(f'Error in saving object to file: {e}')


def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}

        for model_name, model_obj in models.items():

            gs = GridSearchCV(
                model_obj, param_grid=params[model_name], scoring='r2', cv=5, n_jobs=-1
            )

            gs.fit(x_train, y_train)
            model_obj.set_params(**gs.best_params_)
            model_obj.fit(x_train, y_train)

            y_train_pred = model_obj.predict(x_train)
            y_test_pred = model_obj.predict(x_test)

            train_r2_score = r2_score(y_train, y_train_pred)
            test_r2_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_r2_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
