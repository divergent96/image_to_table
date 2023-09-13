'''
Data transformation component

Stores all required functions and code for transformation and cleaning the data
'''

# System and native stuff
import sys
from dataclasses import dataclass
import os

# Standard libs
import numpy as np
import pandas as pd

# Pipeline stuff
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Data scaling and imputation
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Import custom components
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pk1')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            logging.info("Preprocessor configuration Started")

            num_columns = ['writing_score','reading_score']
            cat_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course",]

            num_pipeline = Pipeline(
                steps = [
                ("median_imputer", SimpleImputer(strategy="median")),
                ("std_scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ("freq_imputer",SimpleImputer("most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))# Unsure how this helps
                ]
            )

            logging.info(f"Numerical columns considered: {num_columns}")
            logging.info(f"Categorical columns considered: {cat_columns}")
            
            preprocessor = ColumnTransformer(
                [
                ("num_pipe", num_pipeline, num_columns),
                ("cat_pipe", cat_pipeline, cat_columns)
                
                ]
            )

            logging.info("Preprocessor configuration completed")
        except Exception as e:
            raise CustomException(e, sys)

def transform_data(self, train_path, test_path):

    try:
        logging.info("Data transformation started")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logging.info("Imported train and test sets")

        preprocessing_obj = self.get_data_transformer_object()

        target_col_name = "math_score"
        num_columns = ['writing_score','reading_score']

        input_feat_train_df = train_df.drop(columns = [target_col_name])
        target_feat_train_df = train_df.loc[:,target_col_name]

        input_feat_test_df = test_df.drop(columns = [target_col_name])
        target_feat_test_df = test_df.loc[:,target_col_name]


        logging.info("Start Preprocessing on training and test sets")

        input_feat_train_arr = preprocessing_obj.fit_transform(input_feat_train_df)
        input_feat_test_arr = preprocessing_obj.transform(input_feat_test_df)
        

        train_arr = np.c_[input_feat_train_arr, target_feat_train_df.to_numpy()]
        test_arr = np.c_[input_feat_test_arr, target_feat_test_df.to_numpy()]
        
        logging.info("END Preprocessing train and test sets.")

        logging.info("Saving preprocessed objects")

        save_object(
            file_path = self.data_transformation_config.preprocessor_obj_file_path,
            obj = preprocessing_obj
        )

        logging.info("END preprocessing. Preprocessor has been saved to pickle.")
        
        return(
            train_arr, 
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,
        )
    
    except Exception as e:
        CustomException(e, sys)

        

