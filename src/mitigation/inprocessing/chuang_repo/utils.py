# import torch
# import numpy as np
# from numpy.random import beta
# from sklearn.metrics import average_precision_score
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable

# def sample_batch_sen_idx(X, A, y, batch_size, s):    
#     batch_idx = np.random.choice(np.where(A==s)[0], size=batch_size, replace=False).tolist()
#     batch_x = X[batch_idx]
#     batch_y = y[batch_idx]
#     batch_x = torch.tensor(batch_x).float()
#     batch_y = torch.tensor(batch_y).float()

#     return batch_x, batch_y

# def sample_batch_sen_idx_y(X, A, y, batch_size, s):
#     batch_idx = []
#     for i in range(2):
#         idx = list(set(np.where(A==s)[0]) & set(np.where(y==i)[0]))
#         if batch_size > len(idx):
#             batch_idx += idx
#             batch_idx += np.random.choice(idx, size=len(idx) - batch_size, replace=False).tolist()
#         else:
#             batch_idx += np.random.choice(idx, size=batch_size, replace=False).tolist()

#     batch_x = X[batch_idx]
#     batch_y = y[batch_idx]
#     batch_x = torch.tensor(batch_x).float()
#     batch_y = torch.tensor(batch_y).float()

#     return batch_x, batch_y


# def train_dp(model, criterion, optimizer, X_train, A_train, y_train, method, lam, batch_size=500, niter=100):
#     model.train()
#     for it in range(niter):

#         # Gender Split
#         batch_x_0, batch_y_0 = sample_batch_sen_idx(X_train, A_train, y_train, batch_size, 0)
#         batch_x_1, batch_y_1 = sample_batch_sen_idx(X_train, A_train, y_train, batch_size, 1)

#         if method == 'mixup':
#             # Fair Mixup
#             alpha = 1
#             gamma = beta(alpha, alpha)

#             batch_x_mix = batch_x_0 * gamma + batch_x_1 * (1 - gamma)
#             batch_x_mix = batch_x_mix.requires_grad_(True)

#             output = model(batch_x_mix)

#             # gradient regularization
#             gradx = torch.autograd.grad(output.sum(), batch_x_mix, create_graph=True)[0]

#             batch_x_d = batch_x_1 - batch_x_0
#             grad_inn = (gradx * batch_x_d).sum(1)
#             E_grad = grad_inn.mean(0)
#             loss_reg = torch.abs(E_grad)

#         elif method == 'GapReg':
#             # Gap Regularizatioon
#             output_0 = model(batch_x_0)
#             output_1 = model(batch_x_1)
#             loss_reg = torch.abs(output_0.mean() - output_1.mean())
#         else:
#             # ERM
#             loss_reg = 0

#         # ERM loss
#         batch_x = torch.cat((batch_x_0, batch_x_1), 0)
#         batch_y = torch.cat((batch_y_0, batch_y_1), 0)

#         output = model(batch_x)
#         loss_sup = criterion(output, batch_y)

#         # final loss
#         loss = loss_sup + lam*loss_reg

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


# def train_eo(model, criterion, optimizer, X_train, A_train, y_train, method, lam, batch_size=500, niter=100):
#     model.train()
#     for it in range(niter):

#         # Gender Split
#         batch_x_0, batch_y_0 = sample_batch_sen_idx_y(X_train, A_train, y_train, batch_size, 0)
#         batch_x_1, batch_y_1 = sample_batch_sen_idx_y(X_train, A_train, y_train, batch_size, 1)

#         # separate class
#         batch_x_0_ = [batch_x_0[:batch_size], batch_x_0[batch_size:]]
#         batch_x_1_ = [batch_x_1[:batch_size], batch_x_1[batch_size:]]

#         if method == 'mixup':
#             loss_reg = 0
#             alpha = 1
#             for i in range(2):
#                 gamma = beta(alpha, alpha)
#                 batch_x_0_i = batch_x_0_[i]
#                 batch_x_1_i = batch_x_1_[i]

#                 batch_x_mix = batch_x_0_i * gamma + batch_x_1_i * (1 - gamma)
#                 batch_x_mix = batch_x_mix.requires_grad_(True)
#                 output = model(batch_x_mix)

#                  # gradient regularization
#                 gradx = torch.autograd.grad(output.sum(), batch_x_mix, create_graph=True)[0]
#                 batch_x_d = batch_x_1_i - batch_x_0_i
#                 grad_inn = (gradx * batch_x_d).sum(1)
#                 loss_reg += torch.abs(grad_inn.mean())

#         elif method == "GapReg":
#             loss_reg = 0
#             for i in range(2):
#                 batch_x_0_i = batch_x_0_[i]
#                 batch_x_1_i = batch_x_1_[i]

#                 output_0 = model(batch_x_0_i)
#                 output_1 = model(batch_x_1_i)
#                 loss_reg += torch.abs(output_0.mean() - output_1.mean())
#         else:
#             # ERM
#             loss_reg = 0

#         # ERM loss
#         batch_x = torch.cat((batch_x_0, batch_x_1), 0)
#         batch_y = torch.cat((batch_y_0, batch_y_1), 0)

#         output = model(batch_x)
#         loss_sup = criterion(output, batch_y)

#         # final loss
#         loss = loss_sup + lam*loss_reg

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


# def evaluate_dp(model, X_test, y_test, A_test):
#     model.eval()
#     # calculate average precision
#     X_test_cuda = torch.tensor(X_test).float()
#     output = model(X_test_cuda)
#     return output[:, 1].data.cpu().numpy(), output.data.cpu().numpy()


# def evaluate_eo(model, X_test, A_test):
#     model.eval()
#     X_test_cuda = torch.tensor(X_test).float()
#     output = model(X_test_cuda)
#     return output[:, 1].data.cpu().numpy(), output.data.cpu().numpy()