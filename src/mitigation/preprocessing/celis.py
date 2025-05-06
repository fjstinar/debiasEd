# import numpy as np
# import pandas as pd

# from mitigation.preprocessing.preprocessor import PreProcessor
# from mitigation.preprocessing.fair_max_entropy.domain import Domain
# from mitigation.preprocessing.fair_max_entropy.memory import MemoryTrie
# from mitigation.preprocessing.fair_max_entropy.max_entropy_distribution import MaxEnt
# from mitigation.preprocessing.fair_max_entropy.fair_maximum_entropy import FairMaximumEntropy
# from mitigation.preprocessing.fair_max_entropy.fair_maximum_entropy import reweightSamples
# from mitigation.preprocessing.fair_max_entropy.utils import  *



# ### NEED TO FIND INSTALLATION THAT WORKS
# class CelisPreProcessor(PreProcessor):
#     """Resampling pre-processing

#     References:
#         Celis, L. E., Keswani, V., & Vishnoi, N. (2020, November). Data preprocessing to mitigate bias: A maximum entropy based approach. In International conference on machine learning (pp. 1349-1359). PMLR.
#     """
    
#     def __init__(self, settings: dict):
#         super().__init__(settings)
#         self._name = 'celis et al.'
#         self._notation = 'celis'
#         self._preprocessor_settings = self._settings['preprocessors']['celis']
#         self._information = {}

#     def transform(self, 
#         x_train: list, y_train: list, demo_train: list,
#         ):
#         """
#         Args:
#             x_train (list): training feature data 
#             y_train (list): training label data
#             demo_train(list): training demographics data
#             x_val (list): validation feature data
#             y_val (list): validation label data
#             demo_val (list): validation demographics data
#         """
#         return x_train, y_train, demo_train
    
#     def _get_unique_values(self, x):
#         domainArray = [
#             [uv for uv in np.unique(np.array(x)[:, dimension])] for dimension in range(len(x[0]))
#         ]
#         return domainArray

#     def _format_data(self, x, y, demographics):
#         """Format the arrays as a pandas dataframe, as shown here:
#         https://github.com/vijaykeswani/Fair-Max-Entropy-Distributions/blob/master/Codes/Utils.py
#             x (_type_): features
#             y (_type_): labels
#             demographics (_type_): demographic attributes 
#         """
#         demographic_attributes = self.extract_demographics(demographics)
#         sensitive_attribute = self.get_binary_protected_privileged(demographic_attributes)

#         # features
#         concatenate = [
#             [*x[student], sensitive_attribute[student], y[student]] for student in range(len(x))
#         ]
#         data = np.array(concatenate)

#         # names of the features
#         feature_names = ['f_{}'.format(f_i) for f_i in range(len(x[0]))]
#         all_names = feature_names + ['demographic', 'label']
#         return data, feature_names, all_names

#     def fit_transform(self, 
#             x_train: list, y_train: list, demo_train: list,
#             x_val: list, y_val: list, demo_val: list
#         ):
#         """trains the model and transform the data given the initial training data x, and labels y. 
#         Warning: Init the model every time this function is called

#         Args:
#             x_train (list): training feature data 
#             y_train (list): training label data
#             demo_train(list): training demographics data
#             x_val (list): validation feature data
#             y_val (list): validation label data
#             demo_val (list): validation demographics data
#         """
#         # parameters
#         C = 0.1
#         delta = 0
#         n_samples = len(x_train) * self._preprocessor_settings['sampling_factor']

#         # Data Preparation
#         simple_samples, feature_names, all_names = self._format_data(x_train, y_train, demo_train)
#         domain_array = self._get_unique_values(x_train)
#         domain_array.append([0, 1]) # sensitive attributes possible values
#         simple_domain = Domain(all_names, domain_array)
#         sensitive_attribute = simple_domain.labels.index('demographic')
#         label_index = len(simple_samples[0]) - 1

#         domain = getDomain(domain_array)
#         raw_data_dist = getDistribution(simple_samples, domain) + np.array([0.0000001] * len(domain))
#         utility = getUtility(simple_samples, raw_data_dist, domain)
#         max_entropy = FairMaximumEntropy(
#             simple_domain, simple_samples, C, delta, sensitive_attribute,
#             reweight=True, reweightXindices=[sensitive_attribute],
#             reweightYindices=[label_index], alterMean=True
#         )
#         dataset = max_entropy.sample(n_samples)
#         x_sampled = np.array(dataset)[:, :len(feature_names)]
#         y_sampled = np.array(dataset)[:, -1]
#         demo_sampled = np.array(dataset)[:, -2]
#         return x_sampled, y_sampled, demo_sampled
        
#     def get_information(self):
#         """For each pre-processor, returns information worth saving for future results
#         """
#         return self._information
    
        
