import os
import sys
from sklearn.model_selection import train_test_split
sys.path.insert(0,'D:\SampleProject\src')
from logger import logging
from exception import CustomException
from utils import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    data_transformation_path=os.path.join('artifacts','data_transformation.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation(self):
        try:
            num_features=['reading_score']
            cat_features=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            logging.info("lets create pipelines")

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler())
                ]

            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("ohe",OneHotEncoder())
                ]
            )
            logging.info("numerical and categorical pipeline successfully created")
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_features),
                    ("cat_pipeline",cat_pipeline,cat_features)
                ]
            )
            return preprocessor

        except Exception as e:
            logging.info("error occured in creating data transformation")
            CustomException(e,sys)
   
    def initiate_data_transformation(self,train_path,test_path):
            try:
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)
                target_features='writing_score'
                input_train=train_df.drop(target_features,axis=1)
                input_test=test_df.drop(target_features,axis=1)
                transformation=self.get_data_transformation()
                input_train_tr=transformation.fit_transform(input_train)
                input_test_tr=transformation.transform(input_test)
                train_arr=np.c_[input_train_tr,np.array(train_df[target_features])]
                test_arr=np.c_[input_test_tr,np.array(test_df[target_features])]
                save_object(self.data_transformation_config.data_transformation_path,transformation)

                return (train_arr,test_arr)

            except Exception as e:
                logging.info("ERROR OCCURED IN DATA TRANSFORMATION")
                CustomException(e,sys)