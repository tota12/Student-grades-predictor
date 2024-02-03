import os
import sys
from src.exception import CustomException
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
