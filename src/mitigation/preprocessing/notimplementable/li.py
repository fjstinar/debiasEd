import os
import time
import argparse
import gurobipy as gp
import numpy as np
import pandas as pd
from typing import Sequence

from mitigation.preprocessing.preprocessor import PreProcessor

from mitigation.preprocessing.influence_fairness.dataset import fetch_data, DataTemplate
from mitigation.preprocessing.influence_fairness.eval import Evaluator
from mitigation.preprocessing.influence_fairness.model import LogisticRegression
from mitigation.preprocessing.influence_fairness.fair_fn import grad_ferm, grad_dp, loss_ferm, loss_dp
from mitigation.preprocessing.influence_fairness.utils import fix_seed, save2csv

class LiPreProcessor(PreProcessor):
    """Resampling pre-processing

    References:
        Li, P., & Liu, H. (2022, June). Achieving fairness at no utility cost via data reweighing with influence. In International Conference on Machine Learning (pp. 12917-12930). PMLR.
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'li et al.'
        self._notation = 'li'
        self._preprocessor_settings = self._settings['preprocessors']['li']
        self._information = {}

    def transform(self, 
        x_train: list, y_train: list, demo_train: list,
        ):
        """
        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            demo_train(list): training demographics data
            x_val (list): validation feature data
            y_val (list): validation label data
            demo_val (list): validation demographics data
        """
        raise NotImplementedError

    def lp(fair_infl: Sequence, util_infl: Sequence, fair_loss: float, alpha: float, beta: float,
        gamma: float) -> np.ndarray:
        num_sample = len(fair_infl)
        max_fair = sum([v for v in fair_infl if v < 0.])
        max_util = sum([v for v in util_infl if v < 0.])

        print("Maximum fairness promotion: %.5f; Maximum utility promotion: %.5f;" % (max_fair, max_util))

        all_one = np.array([1. for _ in range(num_sample)])
        fair_infl = np.array(fair_infl)
        util_infl = np.array(util_infl)
        model = gp.Model()
        x = model.addMVar(shape=(num_sample,), lb=0, ub=1)

        if fair_loss >= -max_fair:
            print("=====> Fairness loss exceeds the maximum availability")
            model.addConstr(util_infl @ x <= 0. * max_util, name="utility")
            model.addConstr(all_one @ x <= alpha * num_sample, name="amount")
            model.setObjective(fair_infl @ x)
            model.optimize()
        else:
            model.addConstr(fair_infl @ x <= beta * -fair_loss, name="fair")
            model.addConstr(util_infl @ x <= gamma * max_util, name="util")
            model.setObjective(all_one @ x)
            model.optimize()

        print("Total removal: %.5f; Ratio: %.3f%%\n" % (sum(x.X), (sum(x.X) / num_sample) * 100))

        return 1 - x.X


    def fit_transform(self, 
            x_train: list, y_train: list, demo_train: list,
            x_val: list, y_val: list, demo_val: list
        ):
        """trains the model and transform the data given the initial training data x, and labels y. 
        Warning: Init the model every time this function is called

        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            demo_train(list): training demographics data
            x_val (list): validation feature data
            y_val (list): validation label data
            demo_val (list): validation demographics data
        using: https://github.com/brandeis-machine-learning/influence-fairness/blob/main/main.py
        """
        # data
        demographic_attributes = self.extract_demographics(demo_train)

        # vanilla training
        model = LogisticRegression(l2_reg=self._preprocessor_settings['C_factor'])
        model.fit(x_train, y_train)
        if self._preprocessor_settings['fairness_metric'] == 'eop':
            ori_fair_loss_val = loss_ferm(model.log_loss, x_val, y_val, demo_val)
        elif self._preprocessor_settings['fairness_metric'] == 'dp':
            pred_val, _ = model.pred(x_val)
            ori_fair_loss_val = loss_dp(x_val, demo_val, pred_val)

        ori_util_loss_val = model.log_loss(x_val, y_val)

        # Compute the influence and solve lp
        pred_train, _ = model.pred(x_train)
        train_total_grad, train_indiv_grad = model.grad(x_train, y_train)
        util_loss_total_grad, acc_loss_indiv_grad = model.grad(x_val, y_val)
        if self._preprocessor_settings['fairness_metric'] == 'eop':
            fair_loss_total_grad = grad_ferm(model.grad, x_val, y_val, demo_val)
        elif self._preprocessor_settings['fairness_metric'] == 'dp':
            fair_loss_total_grad = grad_dp(model.grad_pred, x_val, demo_val)

        hess = model.hess(x_train)
        util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)
        fair_grad_hvp = model.get_inv_hvp(hess, fair_loss_total_grad)
        util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
        fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)
        sample_weight = self.lp(
            fair_pred_infl, util_pred_infl, ori_fair_loss_val, 
            args.alpha, args.beta, args.gamma
        )

        # train with weighted samples
        model.fit(data.x_train, data.y_train, sample_weight=sample_weight)
        if args.metric == "eop":
            upd_fair_loss_val = loss_ferm(model.log_loss, data.x_val, data.y_val, data.s_val)
        elif args.metric == "dp":
            pred_val, _ = model.pred(data.x_val)
            upd_fair_loss_val = loss_dp(data.x_val, data.s_val, pred_val)
        else:
            raise ValueError
        upd_util_loss_val = model.log_loss(data.x_val, data.y_val)
        




        raise NotImplementedError
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
