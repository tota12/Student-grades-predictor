import pandas as pd
import os
import sys
from src.exception import CustomException
from src.utils import load_obj


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            # load the model and preprocessor
            model = load_obj(file_path=model_path)
            preprocessor = load_obj(file_path=preprocessor_path)
            print('Model and preprocessor loaded successfully')

            scaled_features = preprocessor.transform(features)
            print('Features scaled successfully')

            prediction = model.predict(scaled_features)
            print('Prediction done')
            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def to_pd(self):
        try:

            data_dict = {
                'gender': [self.gender],
                'race/ethnicity': [self.race_ethnicity],
                'parental level of education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test preparation course': [self.test_preparation_course],
                'reading score': [self.reading_score],
                'writing score': [self.writing_score],
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)
