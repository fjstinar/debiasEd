# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import random
# from pandas.core.frame import DataFrame

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader
# import torch.nn.functional as F

# from sklearn.datasets import fetch_openml
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight

# from fairlearn.metrics import MetricFrame
# import utils
# import argparse

# class PandasDataSet(TensorDataset):

#     def __init__(self, *dataframes):
#         tensors = (self._df_to_tensor(df) for df in dataframes)
#         super(PandasDataSet, self).__init__(*tensors)

#     def _df_to_tensor(self, df):
#         if isinstance(df, np.ndarray):
#             return torch.from_numpy(df).float()
#         return torch.from_numpy(df.values).float()

# class Classifier(nn.Module):

#     def __init__(self, n_features, n_class=2, n_hidden=32, p_dropout=0.2):
#         super(Classifier, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(n_features, n_hidden*2),
#             nn.ReLU(),
#             nn.Dropout(p_dropout),
#             nn.Linear(n_hidden*2, n_hidden),
#             nn.ReLU(),
#             nn.Dropout(p_dropout),
#             nn.Linear(n_hidden, n_class),
#         )

#     def forward(self, x):
#         return self.network(x)

# class Classifier_lr(nn.Module):
#     def __init__(self, n_features, n_class=2):
#         super(Classifier_lr, self).__init__()

#         self.linear = nn.Linear(n_features, n_class)


#     def forward(self, x):
        
#         return self.linear(x)

# def loss_SVM(result, truth, model):
#     truth[truth==0] = -1
#     result = result.squeeze()
#     weight = model.linear.weight.squeeze()

#     loss = torch.mean(torch.clamp(1 - truth * result, min=0))
#     loss += 0.1*torch.mean(torch.mul(weight, weight))

#     return loss


# def CorreErase_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights):
#     for x, y, ind in data_loader:
#         clf.zero_grad()
#         p_y = clf(x)
#         if args.model != 'SVM':
#             loss = criterion(p_y, y.long())
#         else:
#             loss = criterion(p_y, y, clf)

#         for related_attr, related_weight in zip(related_attrs, related_weights):
#             selected_column = list(map(lambda s,related_attr: related_attr in s, processed_X_train.keys(),[related_attr]*len(processed_X_train.keys())))
#             cor_loss = torch.sum(torch.abs(torch.mean(torch.mul(x[:,selected_column].reshape(1,x.shape[0],-1) - x[:,selected_column].mean(dim=0).reshape(1,1,-1), (p_y-p_y.mean(dim=0)).transpose(0,1).reshape((-1,p_y.shape[0],1))),dim=1)))

#             #print('classification loss: {}, feature correlation loss: {}'.format(loss.item(), cor_loss.item()))
#             loss = loss + cor_loss*related_weight

#         loss.backward()
#         optimizer.step()

#     return clf

# ##group fairness loss
# def Gfair_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights):
#     for x, y, ind in data_loader:
#         clf.zero_grad()
#         p_y = clf(x)
#         if args.model != 'SVM':
#             loss = criterion(p_y, y.long())
#         else:
#             loss = criterion(p_y, y, clf)
#         #
#         for related_attr, related_weight in zip(related_attrs, related_weights):
#             group_TPR = utils.groupTPR(p_y, y, np.array(data_frame[related_attr].tolist()), ind)
#             #group_TPR_loss = (group_TPR - (sum(group_TPR)/len(group_TPR)).detach()).sum()*related_weight
#             group_TPR_loss = torch.square(max(group_TPR).detach() - min(group_TPR))

#             #group_TNR = utils.groupTNR(p_y, y, np.array(data_frame[related_attr].tolist()), ind)
#             #group_TNR_loss = torch.square(max(group_TNR).detach() - min(group_TNR))
#             #print('classification loss: {}, group TPR loss: {}, group TNR loss: {}'.format(loss.item(), group_TPR_loss.item(), group_TNR_loss.item()))
#             #print('classification loss: {}, group TPR loss: {}'.format(loss.item(), group_TPR_loss.item()))
#             loss = loss + group_TPR_loss*related_weight
#         loss.backward()
#         optimizer.step()
#     return clf

# #correlation regularization with learned weights
# def CorreLearn_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights, weightSum):
    
#     for x, y, ind in data_loader:
#         UPDATE_MODEL_ITERS = 1
#         UPDATE_WEIGHT_ITERS = 1

#         #update model
#         for iter in range(UPDATE_MODEL_ITERS):
#             clf.zero_grad()
#             p_y = clf(x)
#             if args.model != 'SVM':
#                 loss = criterion(p_y, y.long())
#             else:
#                 loss = criterion(p_y, y, clf)

#             for related_attr, related_weight in zip(related_attrs, related_weights.tolist()):
#                 selected_column = list(map(lambda s,related_attr: related_attr in s, processed_X_train.keys(),[related_attr]*len(processed_X_train.keys())))
#                 cor_loss = torch.sum(torch.abs(torch.mean(torch.mul(x[:,selected_column].reshape(1,x.shape[0],-1) - x[:,selected_column].mean(dim=0).reshape(1,1,-1), (p_y-p_y.mean(dim=0)).transpose(0,1).reshape((-1,p_y.shape[0],1))),dim=1)))

#                 #print('classification loss: {}, feature correlation loss: {}'.format(loss.item(), cor_loss.item()))
#                 loss = loss + cor_loss*related_weight*weightSum

#             loss.backward()
#             optimizer.step()

#         #update weights
#         #ipdb.set_trace()
#         for iter in range(UPDATE_WEIGHT_ITERS):
#             with torch.no_grad():
#                 p_y = clf(x)

#                 cor_losses = []
#                 for related_attr in related_attrs:
#                     selected_column = list(map(lambda s,related_attr: related_attr in s, processed_X_train.keys(),[related_attr]*len(processed_X_train.keys())))
#                     cor_loss = torch.sum(torch.abs(torch.mean(torch.mul(x[:,selected_column].reshape(1,x.shape[0],-1) - x[:,selected_column].mean(dim=0).reshape(1,1,-1), (p_y-p_y.mean(dim=0)).transpose(0,1).reshape((-1,p_y.shape[0],1))),dim=1)))

#                     cor_losses.append(cor_loss.item())

#                 cor_losses = np.array(cor_losses)

#                 cor_order = np.argsort(cor_losses)

#                 #compute -v. represent it as v.
#                 beta = args.beta
#                 v = cor_losses[cor_order[0]]+ 2*beta
#                 cor_sum = cor_losses[cor_order[0]]
#                 l=1
#                 for i in range(cor_order.shape[0]-1):
#                     if cor_losses[cor_order[i+1]] < v:
#                         cor_sum = cor_sum + cor_losses[cor_order[i+1]]
#                         v = (cor_sum+2*beta)/(i+2)
#                         l = l+1
#                     else:
#                         break
                
#                 #compute lambda
#                 for i in range(cor_order.shape[0]):
#                     if i <l:
#                         related_weights[cor_order[i]] = (v-cor_losses[cor_order[i]])/(2*beta)
#                     else:
#                         related_weights[cor_order[i]] = 0



#                 '''
#                 #older optimization version
#                 #update
#                 #related_weights = related_weights - cor_losses*0.001
#                 #mapping
#                 #related_weights[related_weights<0] = 0
#                 #related_weights = related_weights/sum(related_weights)*weightSum
#                 '''


#     return clf, related_weights



    