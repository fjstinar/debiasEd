# from shutil import copytree

# import os
# import logging
# import pickle
# import numpy as np
# import pandas as pd

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from shutil import copytree, rmtree
# from copy import deepcopy
# from typing import Tuple

# from mitigation.inprocessing.chuang_repo.utils import train_dp, train_eo, evaluate_dp, evaluate_eo
# from mitigation.inprocessing.chuang_repo.model import Net
# from mitigation.inprocessing.inprocessor import InProcessor

# class ChuangInProcessor(InProcessor):
#     """inprocessing

#         References:
#             Chuang, C. Y., & Mroueh, Y. (2021). Fair mixup: Fairness via interpolation. arXiv preprint arXiv:2103.06503.
#             https://github.com/chingyaoc/fair-mixup/tree/master/adult
#     """
    
#     def __init__(self, settings: dict):
#         super().__init__(settings)
#         self._name = 'chuang et al.'
#         self._notation = 'chuang'
#         self._inprocessor_settings = self._settings['inprocessors']['chuang']
#         self._information = {}
#         self._fold = -1

#     def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
#         demographic_attributes = self.extract_demographics(demographics)
#         demos = self.get_binary_privileged(demographic_attributes)
#         return np.array(x), np.array(y), np.array(demos)
    
#     def _format_features(self, x:list, demographics:list) -> list:
#         demographic_attributes = self.extract_demographics(demographics)
#         demos = self.get_binary_privileged(demographic_attributes)
#         return np.array(x), np.array(demos)

#     def _init_model(self):
#         """Initiates a model with self._model
#         """
#         self.model = Net(input_size=self._input_size)
#         if self._inprocessor_settings['optimising'] == 'eo':
#             self.train_function = train_eo
#             self.eval_function = evaluate_eo
#         if self._inprocessor_settings['optimising'] == 'dp':
#             self.train_function = train_dp
#             self.eval_function = evaluate_dp
        

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
#         self._input_size = len(x_train[0])

#         self._init_model()
#         optimizer = optim.Adam(self.model.parameters(), lr=1e3)
#         criterion = nn.BCELoss()
#         data, labels, demos = self._format_final(x_train, y_train, demographics_train)
        
#         for ep in range(self._inprocessor_settings['epochs']):
#             self.train_function(
#                 self.model, criterion, optimizer, data, demos, labels,
#                 self._inprocessor_settings['method'], self._inprocessor_settings['lambda']
#             )
    
#     def predict(self, x: list, y, demographics: list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#             return x and y
#         """
#         data, demos = self._format_features(x, demographics)
#         pred, _ = self.eval_function(self.model, data, demos)
#         return pred

#     def predict_proba(self, x: list, demographics:list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#         """
#         # predictions = predictions.cpu().detach().numpy()
#         # pred0 = 1 - np.array(predictions)
#         # probabilities = np.array([predictions, pred0]).reshape(2, len(predictions)).transpose()
#         data, demos = self._format_features(x, demographics)
#         _, probs = self.eval_function(self.model, data, demos)
#         return probs

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
    

        
