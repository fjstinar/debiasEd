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

# from sklearn.metrics import accuracy_score
# from mitigation.inprocessing.alghamdi_repo.GroupFair import GFair
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from mitigation.inprocessing.inprocessor import InProcessor

# class AlghamdiInProcessor(InProcessor):
#     """inprocessing

#         References:
#             Alghamdi, W., Hsu, H., Jeong, H., Wang, H., Michalak, P. W., Asoodeh, S., & Calmon, F. P. (2022). Beyond adult and compas: Fairness in multi-class prediction. arXiv preprint arXiv:2206.07801.
#             https://github.com/HsiangHsu/Fair-Projection/tree/main/fair-projection/adult-compas
#     """
    
#     def __init__(self, settings: dict):
#         super().__init__(settings)
#         self._name = 'alghamdi et al.'
#         self._notation = 'alghamdi'
#         self._inprocessor_settings = self._settings['inprocessors']['alghamdi']
#         self._information = {}
#         self._fold = -1

#     def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
#         return np.array(x), np.array(y), np.array(demographics)
    
#     def _format_features(self, x:list, demographics:list) -> list:
#         return np.array(x)

#     def _init_model(self):
#         """Initiates a model with self._model
#         """
#         if self._inprocessor_settings['model'] == 'gbm':
#             self.model = GradientBoostingClassifier(random_state=self._settings['seeds']['inprocessor'])  # will predict Y from X
#             self._clf_SgX = GradientBoostingClassifier(random_state=self._settings['seeds']['inprocessor'])  # will predict S from X (needed for SP)
#             self._clf_SgXY = GradientBoostingClassifier(random_state=self._settings['seeds']['inprocessor'])  # will predict S from (X,Y)
#         elif self._inprocessor_settings['model'] == 'logit':
#             self.model = LogisticRegression(random_state=self._settings['seeds']['inprocessor'], max_iter=10000)  # will predict Y from X
#             self._clf_SgX = LogisticRegression(random_state=self._settings['seeds']['inprocessor'], max_iter=10000)  # will predict S from X (needed for SP)
#             self._clf_SgXY = LogisticRegression(random_state=self._settings['seeds']['inprocessor'], max_iter=10000)  # will predict S from (X,Y)
#         elif self._inprocessor_settings['model'] == 'rfc':
#             self.model = RandomForestClassifier(random_state=self._settings['seeds']['inprocessor'], n_estimators=10, min_samples_leaf=10)  # will predict Y from X
#             self._clf_SgX = RandomForestClassifier(random_state=self._settings['seeds']['inprocessor'], n_estimators=10, min_samples_leaf=10)  # will predict S from X (needed for SP)
#             self._clf_SgXY = RandomForestClassifier(random_state=self._settings['seeds']['inprocessor'], n_estimators=10, min_samples_leaf=10)  # will pre

#     def init_model(self):
#         self._init_model()

#     def search_threshold(self, clf, X, y, s):
#         thresholds = np.arange(0.0, 1.0, 0.01)
#         acc_score = np.zeros((len(thresholds)))
#         y_prob = np.squeeze(clf.predict_proba(X=X, s=s), axis=2)

#         for i, t in enumerate(thresholds):
#             # Corrected probabilities
#             y_pred = (y_prob[:, 1] > t).astype('int')
#             # Calculate the acc scores
#             acc_score[i] = accuracy_score(y, y_pred)

#         index = np.argmax(acc_score)
#         thresholdOpt = thresholds[index]
#         return thresholdOpt

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
#         x_train, y_train, demographics_train = self._format_final(x_train, y_train, demographics_train)
#         demographic_attributes = self.extract_demographics(demographics_train)
#         demos = np.array(self.get_binary_protected_privileged(demographic_attributes))
#         self.gf = GFair(self.model, self._clf_SgX, self._clf_SgXY)
#         self.gf.fit(X=x_train, y=y_train, s=demos, sample_weight=None)

#         constraints = [(self._inprocessor_settings['constraint'], self._inprocessor_settings['tolerance'])]
#         self.gf.project(X=x_train, s=demos, constraints=constraints, rho=2, max_iter=500, method='tf')
#         self.threshold = self.search_threshold(self.gf, x_train, y_train, demos)
    

#     def predict(self, x: list, y, demographics: list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#             return x and y
#         """
#         preds = self.gf.predict_proba(np.array(x))
#         preds = [1 if p[1] > self.threshold else 0 for p in preds]
#         return preds, y

#     def predict_proba(self, x: list, demographics:list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#         """
#         probas = self.gf.predict_proba(np.array(x))
#         return probas
        
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
    

        
