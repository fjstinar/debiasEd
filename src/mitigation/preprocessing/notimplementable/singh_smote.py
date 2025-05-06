import numpy as np
import pandas as pd
from collections import Counter

from imblearn.over_sampling import SMOTE
from mitigation.preprocessing.preprocessor import PreProcessor

class SinghSmotePreProcessor(PreProcessor):
    """Resampling pre-processing

    References:
        Singh, A., Singh, J., Khan, A., & Gupta, A. (2022). Developing a novel fair-loan classifier through a multi-sensitive debiasing pipeline: Dualfair. Machine Learning and Knowledge Extraction, 4(1), 240-253.
        https://github.com/ariba-k/fair-loan-predictor
        note: self-implementation though the repository exists
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'singsmote et al.'
        self._notation = 'sismo'
        self._preprocessor_settings = self._settings['preprocessors']['singh']
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
        return x_train, y_train, demo_train

    def _get_medians(self, y_train, demographic_attributes):
        counts = {}
        for da in np.unique(demographic_attributes):
            da_indices = [d for d in range(len(demographic_attributes)) if demographic_attributes[d]==da]
            da_ys = [y_train[daidx] for daidx in da_indices]
            da_counter = Counter(da_ys)
            counts[da] = {
                'positive': da_counter[1],
                'negative': da_counter[0]
            }
        positive_medians = np.median([counts[cd]['positive'] for cd in counts])
        negative_medians = np.median([counts[cd]['negative'] for cd in counts])
        return positive_medians, negative_medians

    def _resample(self, x, y, medians):
        smote = SMOTE(sampling_strategy=medians, k_neighbors=self._preprocessor_settings['k'])
        x_res, y_res = smote.fit_resample(x, y)
        return x_res, y_res

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
        positive_medians, negative_medians = self._get_medians(y_train, demographic_attributes)
        medians = {0: int(negative_medians), 1: int(positive_medians)}

        x_sampled, y_sampled = [], []
        for da in np.unique(demographic_attributes):
            # demographic subgroup
            da_indices = [d for d in range(len(demographic_attributes)) if demographic_attributes[d]==da]
            da_xs = [x_train[daidx] for daidx in da_indices]
            da_ys = [y_train[daidx] for daidx in da_indices]

            # sampled
            da_xresampled, da_yresampled = self._resample(da_xs, da_ys, medians)
            x_sampled = x_sampled + da_xresampled
            y_sampled = y_sampled + da_yresampled

        return x_sampled, y_sampled, []
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
