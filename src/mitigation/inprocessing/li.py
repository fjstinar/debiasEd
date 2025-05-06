# import re
# from shutil import copytree

# import os
# import logging
# import pickle
# import numpy as np
# import pandas as pd


# import numpy
# import tensorflow
# from tensorflow import keras
# from keras import backend as K
# from keras.utils import to_categorical
# from keras import layers
# from tensorflow.python.keras.losses import CategoricalCrossentropy, MeanSquaredError, MeanAbsoluteError, Hinge
# from mitigation.inprocessing.li_repo.util_result import get_data_compas
# from mitigation.inprocessing.li_repo.utils_siamese_fair import SiameseFair_ModelCheckpoint, SiameseFair_Multiple
# from shutil import copytree, rmtree
# from copy import deepcopy
# from typing import Tuple

# from mitigation.inprocessing.inprocessor import InProcessor

# class LiInProcessor(InProcessor):
#     """inprocessing

#         References:
#             Li, X., Wu, P., & Su, J. (2023, June). Accurate fairness: Improving individual fairness without trading accuracy. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 12, pp. 14312-14320).
#             https://github.com/Xuran-LI/AccurateFairnessCriterion/tree/main/AccurateFairness/utils
#     """
    
#     def __init__(self, settings: dict):
#         super().__init__(settings)
#         self._name = 'Li et al.'
#         self._notation = 'li'
#         self._inprocessor_settings = self._settings['inprocessors']['li']
#         self._information = {}
#         self._fold = -1

#     def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
#         return np.array(x), np.array(y)
    
#     def _format_features(self, x:list, demographics:list) -> list:
#         return K.constant(np.array(x))

#     def _init_model(self):
#         """Initiates a model with self._model
#         """
#         self.model = None
#         self.model = keras.Sequential()
#         self.model.add(layers.Dense(11, activation="relu"))
#         self.model.add(layers.Dense(2, activation="relu"))
#         self.model.add(layers.Dense(2, activation="softmax"))
#         self.model.build(input_shape=(None, self._inpput_features))
#         self.model.summary()

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
#         self._inpput_features = len(x_train[0])
#         x_train, y_train = self._format_final(x_train, y_train, demographics_train)
#         self._init_model()
#         demographic_attributes = self.extract_demographics(demographics_train)

#         input_list = []
#         output_list = []
#         for j in range(len(x_train)):
#             input_j = keras.Input(shape=(self._inpput_features,), name="features_{}".format(j))
#             output_j = self.model(input_j)
#             input_list.append(input_j)
#             output_list.append(output_j)

#         self.siamese_model = SiameseFair_Multiple(inputs=input_list, outputs=output_list)
#         self.siamese_model.set_parameter(len(x_train), MeanAbsoluteError(), MeanAbsoluteError())
#         self.siamese_model.compile(loss=MeanSquaredError(), optimizer=tensorflow.keras.optimizers.Adam(), metrics=['acc'])
#         history = self.siamese_model.fit(x=x_train, y=y_train, epochs=self._inprocessor_settings['epochs'], 
#         batch_size=self._inprocessor_settings['batch_size'], verbose=0)
#         return history
    
#     def predict(self, x: list, y, demographics: list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#             return x and y
#         """
#         x = self._format_features(x, demographics)
#         return self.siamese_model(x, training=False), y


#     def predict_proba(self, x: list, demographics:list) -> list:
#         """Predict the labels of x

#         Args:
#             x (list): features
            
#         Returns:
#             list: list of raw predictions for each data point
#         """
#         x = self._format_features(x, demographics)
#         self.siamese_model(x, training=False)

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
    

        
