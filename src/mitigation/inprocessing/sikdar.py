# from shutil import copytree

# import os
# import logging
# import pickle
# import numpy as np
# import pandas as pd

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# from sklearn.neural_network import MLPClassifier

# from shutil import copytree, rmtree
# from copy import deepcopy
# from typing import Tuple
# from mitigation.inprocessing.sikdar_repo.Optimizer import TrainerEpisodic, TrainerEpisodicMultiple

# from mitigation.inprocessing.sikdar_repo.OptimizerModel import MetaOptimizerDirection, MetaOptimizerMLP
# from mitigation.inprocessing.sikdar_repo.Rewardfunctions import stat_parity, diff_FPR, diff_FNR, diff_FPR_FNR, diff_Eoppr, diff_Eodd

# from mitigation.inprocessing.inprocessor import InProcessor

# class SikdarInProcessor(InProcessor):
#     """inprocessing

#         References:
#             Sikdar, S., Lemmerich, F., & Strohmaier, M. (2022, June). Getfair: Generalized fairness tuning of classification models. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency (pp. 289-299).
#             https://github.com/Sandipan99/GetFair
#     """
    
#     def __init__(self, settings: dict):
#         super().__init__(settings)
#         self._name = 'sikdar et al.'
#         self._notation = 'sikdar'
#         self._inprocessor_settings = self._settings['inprocessors']['sikdar']
#         self._information = {}
#         self._fold = -1

#     def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
#         demographic_attributes = self.extract_demographics(demographics)
#         return np.array(x), np.array(y)
    
#     def _format_features(self, x:list, demographics:list) -> list:
#         demographic_attributes = self.extract_demographics(demographics)
#         return np.array(x)

#     def _init_model(self):
#         """Initiates a model with self._model
#         """
#         # Choose Model
#         clf = None
#         clf_name = None
#         if self._inprocessor_settings['model_name'] == 'log-reg':
#             clf_name = "logistic regression classifier"
#             self.model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=25000)
#         elif self._inprocessor_settings['model_name'] == 'lin-svm':
#             clf_name = "linear linear SVM classifier"
#             self.model = LinearSVC(loss='squared_hinge', max_iter=125000)
#         elif self._inprocessor_settings['model_name'] == 'mlp':
#             clf_name = "multilayer perceptron"
#             self.model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
#         else:
#             pass

#         # Choose Metric
#         if self._inprocessor_settings['metric_name']=='stpr':
#             self.f_metric = stat_parity
#         elif self._inprocessor_settings['metric_name']=='eoppr':
#             self.f_metric = diff_Eoppr
#         elif self._inprocessor_settings['metric_name']=='eodd':
#             self.f_metric = diff_Eodd
#         else:
#             self.f_metric = None
#         return clf_name


#     def init_model(self):
#         self._init_model()

#     def fit(self, 
#         x_train: list, y_train: list, demographics_train: list,
#         x_val:list, y_val:list, demographics_val: list
#     ):
#         x_train = x_train[:10]
#         y_train = y_train[:10]
#         demographics_train = demographics_train[:10]
#         """fits the model with the training data x, and labels y. 
#         Warning: Init the model every time this function is called

#         Args:
#             x_train (list): training feature data 
#             y_train (list): training label data
#             x_val (list): validation feature data
#             y_val (list): validation label data
#         """
#         clf_name = self._init_model()
#         classifiertype = 'linear' if self._inprocessor_settings['model_name'] in ('log-reg', 'lin-svm') else 'neuralnet'
#         demographic_attributes = self.extract_demographics(demographics_train)
#         demos = self.get_binary_protected_privileged(demographic_attributes)

#         self.model.fit(x_train, y_train)
#         print("PRE")
#         print(np.sum(np.array(self.model.predict(x_train))))
#         meta = MetaOptimizerDirection(
#             hidden_size=self._inprocessor_settings['hidden_size'], 
#             layers=self._inprocessor_settings['layers'],
#             output_size=2
#         )
#         self.trainer = TrainerEpisodic(meta, self.model, [x_train, demos, y_train], self.f_metric, 
#         classifiertype=classifiertype)
#         self.trainer.train(accuracy_threshold=0.55, step_size=0.04, episodes=100)
#         print("POST")
#         ts_pred, score_ = self.trainer.test([x_train, demos, y_train])
#         print(np.sum(np.array(ts_pred)))
    
#     def predict(self, x: list, y, demographics: list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#             return x and y
#         """
#         demographic_attributes = self.extract_demographics(demographics)
#         demos = self.get_binary_protected_privileged(demographic_attributes)
#         preds, _ = self.trainer.test([x, demos, []])
#         return preds

#     def predict_proba(self, x: list, demographics:list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#         """
#         demographic_attributes = self.extract_demographics(demographics)
#         demos = self.get_binary_protected_privileged(demographic_attributes)
#         _, score_ = self.trainer.test([x, demos, []])
#         return score_

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
    

        
