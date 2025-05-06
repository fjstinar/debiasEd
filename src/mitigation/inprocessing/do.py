# from shutil import copytree

# import os
# import logging
# import pickle
# import numpy as np
# import pandas as pd

# # import torch.nn.functional as F
# # import tensorflow as tf
# # import torch
# from shutil import copytree, rmtree
# from copy import deepcopy
# from typing import Tuple
# from mitigation.inprocessing.do_repo.fair_glm import FairGeneralizedLinearModel
# from mitigation.inprocessing.inprocessor import InProcessor

# class DoInProcessor(InProcessor):
#     """inprocessing

#         References:
#             Do, H., Putzel, P., Martin, A. S., Smyth, P., & Zhong, J. (2022, June). Fair generalized linear models with a convex penalty. In International Conference on Machine Learning (pp. 5286-5308). PMLR
#             https://github.com/hyungrok-do/fair-glm-cvx/tree/main/models

#     """
    
#     def __init__(self, settings: dict):
#         super().__init__(settings)
#         self._name = 'do et al.'
#         self._notation = 'do'
#         self._inprocessor_settings = self._settings['inprocessors']['do']
#         self._information = {}
#         self._fold = -1

#     def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
#         data = pd.DataFrame(x)

#         demographic_attributes = self.extract_demographics(demographics)
#         demos = self.get_binary_protected_privileged(demographic_attributes)
#         data['demographics'] = demos

#         return np.array(data), np.array(y)
    
#     def _format_features(self, x:list, demographics:list) -> list:
#         data = pd.DataFrame(x)
#         demographic_attributes = self.extract_demographics(demographics)
#         demos = self.get_binary_protected_privileged(demographic_attributes)
#         data['demographics'] = demos

#         return np.array(data)

#     def _init_model(self):
#         """Initiates a model with self._model
#         """
#         self.model = FairGeneralizedLinearModel(
#             sensitive_index=-1,
#         )

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
#         self._init_model()
#         data, labels = self._format_final(x_train, y_train, demographics_train)

#         self.model.fit(data, labels)
    
#     def predict(self, x: list, y, demographics: list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#             return x and y
#         """
#         data = self._format_features(x, demographics)
#         return self.model._predict(data), y

#     def predict_proba(self, x: list, demographics:list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#         """
#         data = self._format_features(x, demographics)
#         return self.model._predict_proba(data), y

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
    

        
