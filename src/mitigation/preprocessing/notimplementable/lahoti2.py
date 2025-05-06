import numpy as np
import pandas as pd

from mitigation.preprocessing.preprocessor import PreProcessor

class PreProcessor(PreProcessor):
    """Resampling pre-processing

    References:
        
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = ' et al.'
        self._notation = ''
        self._preprocessor_settings = self._settings['preprocessors']['']
        self._information = {}

    def transform(self, 
        x_train: list, y_train: list, demo_train: list,
        ):
        """
        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            demo_train(list): training demographics data
            x_val (list): validation feature data
            y_val (list): validation label data
            demo_val (list): validation demographics data
        """
        raise NotImplementedError

    def fit_transform(self, 
            x_train: list, y_train: list, demo_train: list,
            x_val: list, y_val: list, demo_val: list
        ):
        """trains the model and transform the data given the initial training data x, and labels y. 
        Warning: Init the model every time this function is called

        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            demo_train(list): training demographics data
            x_val (list): validation feature data
            y_val (list): validation label data
            demo_val (list): validation demographics data
        """
        demographic_attributes = self.extract_demographics(demo_train)
        raise NotImplementedError
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
