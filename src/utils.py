import os
import sys
import pickle
import pandas as pd
import numpy as np
from logger import logging
from exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path,obj):
    dir=os.path.dirname(file_path)
    os.makedirs(dir,exist_ok=True)
    with open(file_path,'wb') as path:
        pickle.dump(obj,path)

def evaluate_model(models,X_train,y_train,X_test,y_test):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            model_name=list(models.keys())[i]
            report[model_name]=r2_score(y_test,y_pred)

        return report
    except Exception as e:
        logging.info("ERROR OCCURED DURING MODEL EVALUATION")
        raise CustomException(e,sys)
