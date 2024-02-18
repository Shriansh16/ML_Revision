import os
import sys
sys.path.insert(0,'D:\SampleProject\src')
from logger import logging
from exception import CustomException
from utils import *
import pandas as pd
import numpy as np


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
             model_path='artifacts\model.pkl'
             preprocessor_path='artifacts\data_transformation.pkl'
             model=load_object(model_path)
             preprocessor=load_object(preprocessor_path)
             scaled_data=preprocessor.transform(preprocessor_path)
             predictions=model.predict(scaled_data)
             return predictions
        except Exception as e:
             logging.info("ERROR OCCURED IN PREDICTIONS")
             raise CustomException(e,sys)

class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score

    def get_data_frame(self):
        try:
            Custom_data_input={
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score]
            }
            return pd.DataFrame(Custom_data_input)
        except Exception as e:
            logging.info("error occured during returning the input as dataframe")
            raise CustomException(e,sys)
