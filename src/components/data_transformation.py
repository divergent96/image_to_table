'''
Data transformation component

Stores all required functions and code for transformation and cleaning the data
'''

import sys
from dataclasses import dataclass

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
import os
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pk1')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            num_columns = ['writing_score','reading_score']
        except:
            pass



