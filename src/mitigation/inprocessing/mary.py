# from shutil import copytree

# import os
# import logging
# import pickle
# import numpy as np
# import pandas as pd

# from torch import nn
# import torch.nn.functional as F
# import tensorflow as tf
# import torch
# from shutil import copytree, rmtree
# from copy import deepcopy
# from typing import Tuple

# from mitigation.inprocessing.inprocessor import InProcessor
# from mitigation.inprocessing.mary_repo.utils import evaluate, regularized_learning, chi_squared_l1_kde

# class NetRegression(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(NetRegression, self).__init__()
#         size = 50
#         self.first = nn.Linear(input_size, size)
#         self.fc = nn.Linear(size, size)
#         self.last = nn.Linear(size, num_classes)

#     def forward(self, x):
#         out = F.selu(self.first(x))
#         out = F.selu(self.fc(out))
#         out = self.last(out)
#         return out

# class MaryInProcessor(InProcessor):
#     """inprocessing

#         References:
#             Mary, J., Calauzenes, C., & El Karoui, N. (2019, May). Fairness-aware learning for continuous attributes and treatments. In International Conference on Machine Learning (pp. 4382-4391). PMLR.
#             https://github.com/criteo-research/continuous-fairness

#     """
    
#     def __init__(self, settings: dict):
#         super().__init__(settings)
#         self._name = 'mary et al.'
#         self._notation = 'mary'
#         self._inprocessor_settings = self._settings['inprocessors']['mary']
#         self._information = {}
#         self._fold = -1

#     def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
#         demographic_attributes = self.extract_demographics(demographics)
#         demos = self.get_binary_protected_privileged(demographic_attributes)
#         return np.array(x), np.array(y), np.array(demos)
    
#     def _format_features(self, x:list, demographics:list) -> list:
#         demographic_attributes = self.extract_demographics(demographics)
#         demos = self.get_binary_protected_privileged(demographic_attributes)
#         return np.array(x), np.array(demos)

#     def _init_model(self):
#         """Initiates a model with self._model
#         """
#         self.model = NetRegression(self.input_size, 2)

#     def init_model(self):
#         self._init_model()

#     def fit(self, 
#         x_train: list, y_train: list, demographics_train: list,
#         x_val:list, y_val:list, demographics_val: list
#     ):
#         """fits the model with the training data x, and labels y. 
#         Warning: Init the model every time this function is called

#         Args:
#             x_train (list): training feature data 
#             y_train (list): training label data
#             x_val (list): validation feature data
#             y_val (list): validation label data
#         """
#         self.input_size = len(x_train[0])
#         self._init_model()
#         data, labels, demos = self._format_final(x_train, y_train, demographics_train)

#         self.predict_model = regularized_learning(
#             data, labels, demos, model=self.model, fairness_penalty=chi_squared_l1_kde, 
#             lr=self._inprocessor_settings['lr'], num_epochs=self._inprocessor_settings['epochs']
#         )
    
#     def predict(self, x: list, y, demographics: list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#             return x and y
#         """
#         data, demos = self._format_features(x, demographics)
#         preds = evaluate(self.predict_model, data)
#         return preds[:, 1]

#     def predict_proba(self, x: list, demographics:list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#         """
#         data, demos = self._format_features(x, demographics)
#         preds = evaluate(self.predict_model, data)
#         return preds

#     def save(self, extension='') -> str:
#         """Saving the model in the following path:
#         '../experiments/run_year_month_day/models/model_name_fx.pkl

#         Returns:
#             String: Path
#         """
#         path = '{}/models/'.format(self._settings['experiment']['name'])
#         os.makedirs(path, exist_ok=True)
#         with open('{}{}_{}.pkl'.format(path, self._notation, extension), 'wb') as fp:
#             pickle.dump(self._information, fp)
#         return '{}{}_{}'.format(path, self._notation, extension)

#     def save_fold(self, fold: int) -> str:
#         return self.save(extension='fold_{}'.format(fold))

#     def save_fold_early(self, fold: int) -> str:
#         return self.save(extension='fold_{}_len{}'.format(
#             fold, self._maxlen
#         ))
    

        
