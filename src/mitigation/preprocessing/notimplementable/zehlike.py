from enum import unique
import math
import numpy as np
import pandas as pd

from mitigation.preprocessing.notimplementable.continuous_fairness.cfa import ContinuousFairnessAlgorithm
from mitigation.preprocessing.preprocessor import PreProcessor
from pipelines.crossvalidation_pipeline import CrossValMaker

class ZehlikePreProcessor(PreProcessor):
    """Representation pre-processing

    References:
        Zehlike, M., Hacker, P., & Wiedemann, E. (2020). Matching code and law: achieving algorithmic fairness with optimal transport. Data Mining and Knowledge Discovery, 34(1), 163-200.
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'zehlike et al.'
        self._notation = 'zehlike'
        self._preprocessor_settings = self._settings['preprocessors']['zehlike']
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

    def _format_groups(self, demographics):
        demographic_attributes = self.extract_demographics(demographics)
        unique_attributes = np.unique(demographic_attributes)
        self._demographic_mapping = {
            i: unique_attributes[i] for i in range(len(unique_attributes))
        }
        self._inverse_demographic_mapping = {
            v: k for k, v in self._demographic_mapping.items()
        }
        self._group_names = {
            '[{}]'.format(dm): self._demographic_mapping[dm] for dm in self._demographic_mapping
        }

        groups = pd.DataFrame([ua for ua in range(len(unique_attributes))])
        groups.columns = ['demographic']
        return groups, self._group_names


    def _format_data(self, x, y, demographics):
        """Format the arrays as a pandas dataframe, as shown here:
        https://github.com/MilkaLichtblau/ContinuousFairness/blob/master/data/LSAT/allRace/allEthnicityLSAT.csv

            x (_type_): features
            y (_type_): labels
            demographics (_type_): demographic attributes 
        """
        demographic_attributes = self.extract_demographics(demographics)
        concatenate = [
            [*x[i], y[i], self._inverse_demographic_mapping[demographic_attributes[i]]] for i in range(len(x))
        ]
        data = pd.DataFrame(concatenate)
        data['student'] = [str(i) for i in range(len(data))]
        columns = ['f_{}'.format(f_i) for f_i in range(len(x[0]))] + ['label', 'demographic', 'student']
        data.columns = columns
        return data

    def _compute_score_stepsize(self, ytrain):
        ys = [ytt for ytt in np.unique(ytrain)]
        ys.sort()

        by = [0] + ys[:-1]
        steps = (np.array(ys) - np.array(by))[1:]
        return int(np.min(steps))

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
        groups, group_names = self._format_groups(demo_train)
        data = self._format_data(x_train, y_train, demo_train)
        thetas = np.array([self._preprocessor_settings['thetas'] for _ in range(len(self._group_names))])

        # check that we have a theta for each group
        if groups.shape[0] != len(thetas):
            raise ValueError(
                "invalid number of thetas, should be {numThetas} Specify one theta per group.".format(numThetas=groups.shape[0]))

        regForOT = 5e-3
        score_stepsize = self._compute_score_stepsize(y_train)

        cfa = ContinuousFairnessAlgorithm(
            data, groups, group_names,
            'label', score_stepsize, thetas, regForOT,
            path=self._settings['experiment']['name'], plot=False
        )
        massaged_y = cfa.run()
        print(massaged_y)
        return x_train, massaged_y, demo_train
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
