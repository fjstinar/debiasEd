# from shutil import copytree

# import os
# import logging
# import pickle
# import numpy as np
# import pandas as pd

# # import torch.nn.functional as F
# # import tensorflow as tf
# # import torch
# from sklearn.preprocessing import StandardScaler
# from shutil import copytree, rmtree
# from copy import deepcopy
# from typing import Tuple

# from mitigation.inprocessing.inprocessor import InProcessor

# class ZhaoInProcessor(InProcessor):
#     """inprocessing

#         References:
#             Zhao, T., Dai, E., Shu, K., & Wang, S. (2022, February). Towards fair classifiers without sensitive attributes: Exploring biases in related features. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining (pp. 1433-1442).
#             https://github.com/TianxiangZhao/fairlearn

#     """
    
#     def __init__(self, settings: dict):
#         super().__init__(settings)
#         self._name = 'zhao et al.'
#         self._notation = 'zhao'
#         self._inprocessor_settings = self._settings['inprocessors']['zhao']
#         self._information = {}
#         self._fold = -1

#     def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
#         self.scaler = StandardScaler().fit(x)
#         data = self.scaler.transform(x)
#         return data, np.array(y)
    
#     def _format_features(self, x:list, demographics:list) -> list:
#         data = self.scaler.transform(x)
#         return data

#     def _init_model(self):
#         """Initiates a model with self._model
#         """
#         if self._inprocessor_settings['zhao'] == 'MLP':
#             clf = Classifier(n_features=self.input_size, n_hidden=32,n_class=2)
        
#         self.pre_model = 
#         self.model = 

#     def init_model(self):
#         self._init_model()

#     def pretrain_classifier(self, clf, data_loader, optimizer, criterion):
#         for x, y,_ in data_loader:
#             clf.zero_grad()
#             p_y = clf(x)
#             if self._inprocessor_settings['premodel'] != 'SVM':
#                 loss = criterion(p_y, y.long())
#             else:
#                 loss = criterion(p_y, y, clf)
#             loss.backward()
#             optimizer.step()
#         return clf

#     def Perturb_train(self, clf, data_loader, optimizer, criterion, related_attrs, related_weights):
#         for x, y, ind in data_loader:
#             clf.zero_grad()
#             p_y = clf(x)
#             if self._inprocessor_settings['premodel'] != 'SVM':
#                 loss = criterion(p_y, y.long())
#             else:
#                 loss = criterion(p_y, y, clf)
            
#             for related_attr, related_weight in zip(related_attrs, related_weights):
#                 #x_new = utils.counter_sample(data.data.iloc[ind.int()], related_attr, scaler)
#                 x_new = utils.counter_sample(data.data, ind.int(), related_attr, scaler)
#                 p_y_new = clf(x_new)

#                 #cor_loss = torch.square(p_y[:,1] - p_y_new[:,1]).mean()
#                 p_stack = torch.stack((p_y[:,1], p_y_new[:,1]), dim=1)
#                 p_order = torch.argsort(p_stack,dim=-1)
#                 cor_loss = torch.square(p_stack[:,p_order[:,1].detach()] - p_stack[:,p_order[:,0]]).mean()

#                 #print('classification loss: {}, feature correlation loss: {}'.format(loss.item(), cor_loss.item()))
#                 loss = loss + cor_loss*related_weight

#             loss.backward()
#             optimizer.step()

#         return clf

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
#         demographic_attributes = self.extract_demographics(demographics_train)
#         raise NotImplementedError
    
#     def predict(self, x: list, y, demographics: list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#             return x and y
#         """
#         raise NotImplementedError

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
#         raise NotImplementedError

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
    

        
