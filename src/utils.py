import os
import sys
import pickle
import pandas as pd
import numpy as np


def save_object(file_path,obj):
    dir=os.path.dirname(file_path)
    os.makedirs(dir,exist_ok=True)
    with open(file_path,'wb') as path:
        pickle.dump(obj,path)