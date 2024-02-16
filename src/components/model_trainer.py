import os
import sys
sys.path.insert(0,'D:\SampleProject\src')
from logger import logging
from exception import CustomException
from utils import *
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
            models={"linear_regressor":LinearRegression(),"Random_forest_regressor":RandomForestRegressor(),
                    "gradient_boost":GradientBoostingRegressor(),"adaboost":AdaBoostRegressor(),'knn':KNeighborsRegressor(),
                    "decision_tree":DecisionTreeRegressor()}
            report=evaluate_model(models,X_train,y_train,X_test,y_test)
        except Exception as e:
            logging.info("ERROR OCCURED IN MODEL TRAINING")
            raise CustomException(e,sys)

