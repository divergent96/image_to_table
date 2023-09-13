'''
Training module for model.

Will contain all the models and their components that we want to train.
'''

# System libs
import os
import sys
from dataclasses import dataclass

# Basic imports
import numpy as np
import pandas as pd

# Modelling
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score

# Custom Libs
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts","model.pk1")

class ModelTrainer:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()

    def initiate_model_training(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info("Split input data into training and test sets")

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "XGB": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),    
            }

            model_report:dict = evaluate_models(X_train = X_train,
                                                y_train = y_train,
                                                X_test = X_test,
                                                y_test = y_test,
                                                models = models)
            
            # Get best model and score
            best_model_score = max(sorted(model_report.values()))
            best_model_index = list(model_report.values()).index(best_model_score)
            best_model_name = list(model_report.keys())[best_model_index]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found: {best_model_name}")

            save_object(
                file_path = self.model_training_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_score = r2_score(y_test, predicted)

            return r2_score
        
        except Exception as e:
            raise CustomException(e,sys)