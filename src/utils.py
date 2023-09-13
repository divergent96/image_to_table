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
    
