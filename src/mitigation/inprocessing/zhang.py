# from shutil import copytree

# import os
# import logging
# import pickle
# import numpy as np
# import pandas as pd

# from shutil import copytree, rmtree
# from copy import deepcopy
# from typing import Tuple
# import torch
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split, GridSearchCV, KFold
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.metrics import accuracy_score
# from mitigation.inprocessing.zhang_repo.AdversarialDebiasing import AdversarialDebiasing

# from mitigation.inprocessing.inprocessor import InProcessor

# class ZhangInProcessor(InProcessor):
#     """inprocessing

#         References:
#             Zhang, B. H., Lemoine, B., & Mitchell, M. (2018, December). Mitigating unwanted biases with adversarial learning. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society (pp. 335-340).
#             https://github.com/giandos200/Zhang-et-al-Mitigating-Unwanted-Biases-with-Adversarial-Learning

#     """
    
#     def __init__(self, settings: dict):
#         super().__init__(settings)
#         self._name = 'zhang et al.'
#         self._notation = 'zhang'
#         self._inprocessor_settings = self._settings['inprocessors']['zhang']
#         self._information = {}
#         self._fold = -1

#     def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
#         y=pd.DataFrame(y)
#         y['demographics'] = self.get_binary_protected_privileged(self.extract_demographics(demographics))
#         return np.array(x), y
    
#     def _format_features(self, x:list, demographics:list) -> list:
#         return np.array(x)

#     def _init_model(self):
#         """Initiates a model with self._model
#         """
#         self.model = AdversarialDebiasing(
#             prot_attr='demographics',
#             adversary_loss_weight=0.1, num_epochs=self._inprocessor_settings['epochs'], 
#             batch_size=256, classifier_num_hidden_units=256, random_state=42
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
#         x_train, y_train = self._format_final(x_train, y_train, demographics_train)
#         self._init_model()
#         self.model.fit(x_train, y_train)
    
#     def predict(self, x: list, y, demographics: list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#             return x and y
#         """
#         x = self._format_features(x, demographics)
#         return self.model.predict(x), y

#     def predict_proba(self, x: list, demographics:list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#         """
#         x = self._format_features(x, demographics)
#         predictions = self.model.predict_proba(x)
#         return self.model.predict_proba(x)

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
    

        
