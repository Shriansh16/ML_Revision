import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0,'D:\SampleProject\src')
from logger import logging
from exception import CustomException
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_datapath=os.path.join('artifacts','train.csv')
    test_datapath=os.path.join('artifacts','test.csv')
    raw_datapath=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("initiating data ingestion")
        try:
            df=pd.read_csv('notebook\stud1.csv')
            df.drop(columns=['math_score'],inplace=True)
            logging.info('dataset taken as df')
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_datapath),exist_ok=True)
            logging.info("dividing dataset into train and test")
            train_dataset,test_dataset=train_test_split(df,test_size=0.20,random_state=42)
            logging.info("dataset is successfully divided")
            logging.info("saving different dataset")
            train_dataset.to_csv(self.data_ingestion_config.train_datapath,index=False,header=True)
            test_dataset.to_csv(self.data_ingestion_config.test_datapath,index=False,header=True)
            df.to_csv(self.data_ingestion_config.raw_datapath,index=False,header=True)
            logging.info("all the datasets are saved successfully")
            return(
                self.data_ingestion_config.train_datapath,
                self.data_ingestion_config.test_datapath
            )


        except Exception as e:
            logging.info("error occured in data ingestion")
            raise CustomException(e,sys)



