# import torch
# from torch import nn
# import numpy as np
# import torch.utils.data as data_utils

# from mitigation.inprocessing.mary_repo.pytorch_kde import kde
# from mitigation.inprocessing.mary_repo.hgr import chi_2, hgr_cond

# def regularized_learning(x_train, y_train, z_train, model, fairness_penalty, lr=1e-5, num_epochs=10):
#     # wrap dataset in torch tensors
#     Y = torch.tensor(y_train.astype(np.float32))
#     X = torch.tensor(x_train.astype(np.float32))
#     Z = torch.tensor(z_train.astype(np.float32))
#     dataset = data_utils.TensorDataset(X, Y, Z)
#     dataset_loader = data_utils.DataLoader(dataset=dataset, batch_size=200, shuffle=True)

#     # mse regression objective
#     data_fitting_loss = nn.MSELoss()

#     # stochastic optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

#     for j in range(num_epochs):
#         for i, (x, y, z) in enumerate(dataset_loader):
#             def closure():
#                 optimizer.zero_grad()
#                 outputs = model(x).flatten()
#                 loss = data_fitting_loss(outputs, y)
#                 loss += fairness_penalty(outputs, z)
#                 loss.backward()
#                 return loss

#             optimizer.step(closure)
#     return model

# def chi_squared_l1_kde(X, Y):
#     return chi_2(X, Y, kde)

# def evaluate(model, x):
#     X = torch.tensor(x.astype(np.float32))

#     prediction = model(X).detach().flatten()
#     return prediction