# import numpy as np
# import pandas as pd

# from mitigation.postprocessing.aswathi_repo.equalized_odds import *
# from predictors.predictor import Predictor
# from mitigation.postprocessing.postprocessor import PostProcessor

# class AwasthiPostProcessor(PostProcessor):
#     """post-processing
#     only for logistic regression
#     barycenters and wasserstein distance
#     only to check whether the equalised odds could be harmful in case of corrupted attributes

#     References:
#         Awasthi, P., Kleindessner, M., & Morgenstern, J. (2020, June). Equalized odds postprocessing under imperfect group information. In International conference on artificial intelligence and statistics (pp. 1770-1780). PMLR.
#         https://github.com/matthklein/equalized_odds_under_perturbation/blob/master/experiments_real_data.py
#     """
    
#     def __init__(self, settings: dict):
#         super().__init__(settings)
#         self._name = 'awasthi et al.'
#         self._notation = 'awasthi'
#         self._postprocessor_settings = self._settings['postprocessors']['awasthi']
#         self._information = {}

#     def transform(
#             self, model: Predictor, features:list, ground_truths: list, predictions: list,
#             probabilities: list, demographics: list
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
#         raise NotImplementedError
        
#     def flip_0_11(self, labels):
#         return [-1 if l==0 else 1 for l in labels]

#     def determine_perturbation(self):
#         if (self._postprocessor_settings['perturbation_type']==0):
#             self.perturbation_range = np.arange(0, 1.025, 0.025)

#         if (self._postprocessor_settings['perturbation_type']==1):
#             self.perturbation_range = np.arange(0, 0.525, 0.0125)

#         if (self._postprocessor_settings['perturbation_type']==2):
#             self.perturbation_range = np.arange(0, 1.025, 0.025)

#         if (self._postprocessor_settings['perturbation_type']==3):
#             self.perturbation_range = np.arange(0, 0.525, 0.0125)



#     def perturb_a(a,gamma0,gamma1):
#         # Type 0
#         #flips a=0 to a=1 with probability gamma0 and a=1 to a=0 with probability gamma1
#         a_pert=np.copy(a)
#         a0=(a==0)
#         a1=(a==1)
#         flip_indi = np.random.choice([0, 1], size=sum(a0), p=[1 - gamma0, gamma0])
#         a_pert[a0] = ((1 - a_pert[a0]) ** flip_indi) * (a_pert[a0] ** (1 - flip_indi))
#         flip_indi = np.random.choice([0, 1], size=sum(a1), p=[1 - gamma1, gamma1])
#         a_pert[a1] = ((1 - a_pert[a1]) ** flip_indi) * (a_pert[a1] ** (1 - flip_indi))
#         return a_pert

#     def perturb_a_based_on_score(a,r,score):
#         # Type 1
#         # flips a to 1-a whenever |score-0.5|<=r
#         a_pert=np.copy(a)
#         flip_indi = np.abs(score-0.5)<=r
#         a_pert[flip_indi]=1-a_pert[flip_indi]
#         return a_pert

#     def perturb_a_depending_on_Ytilde_and_Y(a,gamma0,gamma1,y_tilde,y_true):
#         # Type 2
#         # flips a=0 to a=1 with probability gamma0 and a=1 to a=0 with probability gamma1 ONLY IF Ytilde!=Y
#         a_pert=np.copy(a)
#         a0=np.logical_and(a==0,y_tilde!=y_true)
#         a1=np.logical_and(a==1,y_tilde!=y_true)
#         flip_indi = np.random.choice([0, 1], size=sum(a0), p=[1 - gamma0, gamma0])
#         a_pert[a0] = ((1 - a_pert[a0]) ** flip_indi) * (a_pert[a0] ** (1 - flip_indi))
#         flip_indi = np.random.choice([0, 1], size=sum(a1), p=[1 - gamma1, gamma1])
#         a_pert[a1] = ((1 - a_pert[a1]) ** flip_indi) * (a_pert[a1] ** (1 - flip_indi))
#         return a_pert

#     def perturb_a_based_on_score_and_Ytilde(a,r,score,y_true):
#         # Type 3
#         # flips a to 1-a whenever |score-0.5|<=r AND Ytilde!=Y
#         a_pert=np.copy(a)
#         flip_indi = np.logical_and((np.abs(score-0.5)<=r),(np.sign(np.sign(score-0.5)+0.01)!=y_true))
#         a_pert[flip_indi]=1-a_pert[flip_indi]
#         return a_pert

#     def fit_transform( 
#             self, model: Predictor, features:list, ground_truths: list, predictions: list,
#             probabilities: list, demographics: list
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
#         demographic_attributes = self.extract_demographics(demographics)

#         bias_Y1_array=np.zeros((self._postprocessor_settings['n_replicates'], self.perturbation_range.size))
#         bias_Ym1_array=np.zeros((self._postprocessor_settings['n_replicates'], self.perturbation_range.size))
#         bias_Y1_original_array=np.zeros((self._postprocessor_settings['n_replicates'], self.perturbation_range.size))
#         bias_Ym1_original_array=np.zeros((self._postprocessor_settings['n_replicates'], self.perturbation_range.size))
#         error_array=np.zeros((self._postprocessor_settings['n_replicates'], self.perturbation_range.size))
#         error_original_array=np.zeros((self._postprocessor_settings['n_replicates'], self.perturbation_range.size))
#         independence_measure_array=np.zeros((self._postprocessor_settings['n_replicates'], self.perturbation_range.size))

#         for ell in range(self._postprocessor_settings['n_replicates']):
#             for rrr,gamma in enumerate(self.perturbation_range):
#                 if self._postprocessor_settings['perturbation_type']==0:
#                     a_pert = self.perturb_a(demographic_attributes, gamma, gamma)
#                 if self._postprocessor_settings['perturbation_type']==1:
#                     a_pert = self.perturb_a_based_on_score(demographic_attributes, gamma, probabilities)
#                 if self._postprocessor_settings['perturbation_type'] == 2:
#                     a_pert = self.perturb_a_depending_on_Ytilde_and_Y(demographic_attributes, gamma, gamma, self.flip_0_11(predictions), ground_truths)
#                 if self._postprocessor_settings['perturbation_type']==3:
#                     a_pert = self.perturb_a_based_on_score_and_Ytilde(demographic_attributes, gamma, probabilities, ground_truths)


#                 independence_measure_array[ell,rrr] = measure_cond_independence(np.sign(np.sign(probabilities-0.5)+0.01), a_pert, ground_truths, demographic_attributes)

#                 eq_odd_pred_test=equalized_odds_pred(train_data[:,0], np.sign(np.sign(train_data[:,2]-0.5)+0.01), a_pert, np.sign(np.sign(test_data[:,2]-0.5)+0.01),test_data[:,1])
#                 EO_error,EO_biY1,EO_biYm1=compute_error_and_bias(test_data[:,0],eq_odd_pred_test,test_data[:,1])

#                 givenCl_error, givenCl_biY1, givenCl_biYm1=compute_error_and_bias(test_data[:,0],np.sign(np.sign(test_data[:,2]-0.5)+0.01),test_data[:,1])

#                 bias_Y1_array[ell,rrr]=EO_biY1
#                 bias_Ym1_array[ell,rrr]=EO_biYm1
#                 bias_Y1_original_array[ell,rrr]=givenCl_biY1
#                 bias_Ym1_original_array[ell,rrr]=givenCl_biYm1
#                 error_array[ell,rrr]=EO_error
#                 error_original_array[ell,rrr]=givenCl_error

#         raise NotImplementedError
        
#     def get_information(self):
#         """For each pre-processor, returns information worth saving for future results
#         """
#         return self._information
    
        
