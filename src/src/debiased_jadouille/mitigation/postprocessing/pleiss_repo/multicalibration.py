import numpy as np
from functools import reduce
import random

"""https://github.com/sanatonek/fairness-and-callibration/blob/master/multicalib/multicalibration.py
"""
random.seed(1945)

    
sensitive_features = {
                      # 1:age_u30, 65: race_black, 66: female
                      'income':[1, 65, 66],
                      # 2: sex, 5: age
                      'credit': [2, 5],
                      # 3: Age>45, 4: Age<25, 5: Black, 6: Asian, 7: Hispanic, 10: Female
                      'recidivism': [5, 10]
                     }


def calibrate(data, labels, predictions, sensitive_features, alpha, lmbda):
    # calibrated_predictions = predictions.copy()
    calibrated_predictions = np.zeros(predictions.shape) + 0.5
    v_range = np.arange(0,1,1./lmbda)
    for sensitive_feature in sensitive_features:
        # Find the two subset of the sensitive feature
        sensitive_set = [i for i in range(len(data)) if data[i, sensitive_feature] == 1]
        sensitive_set_not = list(set(range(len(data))) - set(sensitive_set))
        for S in [sensitive_set, sensitive_set_not]:
            change = 1
            while change > 0:
                change = 0
                for v in v_range:
                    S_v = [i for i in S if calibrated_predictions[i]<v+(1./lmbda) and calibrated_predictions[i]>=v]
                    if len(S_v)< alpha*lmbda*len(S):
                        continue
                    E_predictions = np.mean(calibrated_predictions[S_v])    # V_hat
                    r = oracle(S_v, E_predictions, alpha/4, labels)
                    if r!=100:
                        calibrated_predictions[S_v] = calibrated_predictions[S_v] + (r-E_predictions)

                    if (calibrated_predictions[S_v]<0).any() or (calibrated_predictions[S_v]>1).any():
                        calibrated_predictions[S_v] = normalize(calibrated_predictions[S_v])
                    if set(S_v)!=set([i for i in S if calibrated_predictions[i] < v + (1. / lmbda) and calibrated_predictions[i] >= v]):
                        change += 1
    return calibrated_predictions


def multicalibrate(data, labels, predictions, sensitive_features, alpha, lmbda, oracle):
    # calibrated_predictions = predictions.copy()
    calibrated_predictions = np.zeros(predictions.shape) + 0.5
    v_range = np.arange(0,1,1./lmbda)
    multicalibrate_sets = all_subsets(data, sensitive_features)
    # criteria set to 1000 if dataset has 113 features else 100 (2 datasets for now...)
    leaf_node_crit = [1000.0 if data.shape[1]==113 else 100.0][0] 
    change = 1
    while change > 0:
        change = 0
        for sets in multicalibrate_sets:
            set_not = list(set(range(len(data))) - set(sets))
            for S in [sets, set_not]:
                # print(data.shape)
                #if len(S)==0: # sk-debug
                if len(S)<leaf_node_crit:
                    continue
                for v in v_range:
                    S_v = [i for i in S if calibrated_predictions[i]<(v+(1./lmbda)) and calibrated_predictions[i]>=v]
                    if len(S_v) <= alpha*lmbda*len(S):
                        continue
                    # print(alpha*lmbda*len(S), "%%%%", len(S_v))
                    E_predictions = np.mean(calibrated_predictions[S_v])    # V_hat
                    r = oracle(S_v, E_predictions, alpha/4, labels)
                    if r!=100:
                        calibrated_predictions[S_v] = calibrated_predictions[S_v] + (r-E_predictions)
                        change += 1

                    if (calibrated_predictions[S_v]<0).any() or (calibrated_predictions[S_v]>1).any():
                        calibrated_predictions[S_v] = normalize(calibrated_predictions[S_v])
                        # if set(S_v)!=set([i for i in S if calibrated_predictions[i] < (v+(1./lmbda)) and calibrated_predictions[i] >= v]):
                        #     change += 1
    return calibrated_predictions


def normalize(x):
    if min(x)==max(x):
        return x/min(x)
    else:
        return (x-min(x))/(max(x)-min(x))


def all_subsets(data, sensitive_features):# all possible subsets of data with all 2^x values of sensitive feature set
    all_subsets =[]
    x = len(sensitive_features)
    for i in range(1 << x):
        current_subset = []
        for j in range(x):
            if((i>>j)&1):
                current_subset.append(j) 
        all_subsets.append(current_subset) 
    target_sets=[]
    for i in range(1,len(all_subsets)):
        required_sensitive_features = np.array(sensitive_features)[all_subsets[i]]
        data_indices_matching_features = [sub_ind for sub_ind in range(len(data)) if (data[sub_ind, list(required_sensitive_features)] == 1).all()]
        target_sets.append(data_indices_matching_features)
    return target_sets
