from src.mitigation.preprocessing.preprocessor import PreProcessor
import numpy as np

class Reweighing(PreProcessor):
    '''
    Reference:
    Kamiran, F., Calders, T. Data preprocessing techniques for classification without discrimination. 
    Knowl Inf Syst 33, 1–33 (2012). https://doi.org/10.1007/s10115-011-0463-8
    '''

    # def __init__(self, ...):
        
        # raise NotImplementedError

    def fit_transform(self, x_train, y_train, demographics_train):

        # computing instance-level weights
        # for s in self.sensitive_features:
        #     for c in set(y_train):



        raise NotImplementedError