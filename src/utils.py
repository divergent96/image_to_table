'''
Custom utilities that support the components required.
'''

# System and native stuff
import sys
from dataclasses import dataclass
import os

# Standard libs
import numpy as np
import pandas as pd

# Storage libs
import dill
import pickle

# Sklearn
from sklearn.metrics import r2_score

# Import custom components
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        dir_path = os.path.dirname(file_path)
        
        with open(file_path,"rb") as file_obj:
            pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]

            model.fit(X_train, y_train)
            
            
            # Generate predications and score
             
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)
