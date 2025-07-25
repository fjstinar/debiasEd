from genetic_algorithm import ContinuousGenAlgSolver
from data import load_ori_population, simulate_dataset
import unfairness_metrics, argparse

from sklearn import metrics, model_selection, pipeline, preprocessing, linear_model, ensemble

import numpy as np
import pandas as pd


class calculation():
    def __init__(self, dataset, labels, groups, index, groupkfold=None):
        self.data = dataset
        self.labels = labels
        self.groups = groups
        self.groupkfold = groupkfold
        self.unfair_index = index
        self.last_unfair = 0
        self.iter = 1
        self.inital_lr = self.initial_calculate(dataset, linear_model.LogisticRegression(max_iter=200, random_state=11798))
        self.inital_rf = self.initial_calculate(dataset, ensemble.RandomForestClassifier(random_state=11798))
        self.inital_et = self.initial_calculate(dataset, ensemble.ExtraTreesClassifier(random_state=11798))

    def initial_calculate(self, dataset, clf):
        if self.groupkfold is None:
            xval = model_selection.KFold(4, shuffle=True, random_state=11798)
        else:
            xval = model_selection.GroupKFold(4)
        # clf = linear_model.LogisticRegression(max_iter=200, random_state=11798)
        # clf = ensemble.RandomForestClassifier(random_state=11798)

        scoring = {}
        for m in unfairness_metrics.UNFAIRNESS_METRICS:
            if m == "calibration":
                metric = unfairness_metrics.UnfairnessMetric(pd.Series(self.groups), m)
                scoring[m] = metrics.make_scorer(metric, needs_proba=True)
            else:
                metric = unfairness_metrics.UnfairnessMetric(pd.Series(self.groups), m)
                scoring[m] = metrics.make_scorer(metric)
        scoring['auc'] = metrics.make_scorer(metrics.roc_auc_score)
        scoring['acc'] = metrics.make_scorer(metrics.accuracy_score)
        pipe = pipeline.Pipeline([
            ('standardize', preprocessing.StandardScaler()),
            ('model', clf)
        ])
        if self.groupkfold is None:
            result = model_selection.cross_validate(pipe, dataset, pd.Series(self.labels), verbose=0, cv=xval,
                                                    scoring=scoring
                                                    , return_estimator=True)
        else:
            result = model_selection.cross_validate(pipe, dataset, pd.Series(self.labels), verbose=0,
                                                    cv=xval, groups=self.groupkfold,
                                                    scoring=scoring
                                                    , return_estimator=True)
        unfair_score = []
        unfair_score.append(result['test_' + unfairness_metrics.UNFAIRNESS_METRICS[self.unfair_index]].mean())
        unfair_score.append(result['test_auc'].mean())
        unfair_score.append(result['test_acc'].mean())
        return unfair_score[0]

    def calculate_corr(self, dataset):
        corr = np.corrcoef(dataset.T)
        pos = np.where(np.abs(corr) > 0.4)
        return pos, corr[pos]

    def fit_scores(self, population, labels, idx_groups, gen_percent, expect_score, pnt=False):
        score, unfair_scores = [], []
        for i in range(population.shape[0]):
            self.coeff = 1
            if self.groupkfold is not None:
                xval = model_selection.GroupKFold(4)
            else:
                xval = model_selection.KFold(4, shuffle=True, random_state=11798)

            clf = linear_model.LogisticRegression(max_iter=200, random_state=11798)
            # clf = ensemble.RandomForestClassifier(random_state=11798)
            # clf = ensemble.ExtraTreesClassifier(random_state=11798)
            sml = np.count_nonzero(np.equal(self.data, population[i]) == 1) / (self.data.shape[0]*self.data.shape[1])

            scoring = {}
            for m in unfairness_metrics.UNFAIRNESS_METRICS:
                if m == "calibration":
                    metric = unfairness_metrics.UnfairnessMetric(pd.Series(self.groups), m)
                    scoring[m] = metrics.make_scorer(metric, needs_proba=True)
                else:
                    metric = unfairness_metrics.UnfairnessMetric(pd.Series(self.groups), m)
                    scoring[m] = metrics.make_scorer(metric)
            scoring['auc'] = metrics.make_scorer(metrics.roc_auc_score)
            scoring['acc'] = metrics.make_scorer(metrics.accuracy_score)

            pipe = pipeline.Pipeline([
                ('standardize', preprocessing.StandardScaler()),
                ('model', clf),
            ])

            if self.groupkfold is not None:
                result = model_selection.cross_validate(pipe, population[i], pd.Series(labels), verbose=0,
                                                        cv=xval, groups=self.groupkfold,
                                                        scoring=scoring
                                                        , return_estimator=True)
            else:
                result = model_selection.cross_validate(pipe, population[i], pd.Series(labels), verbose=0,
                                                        cv=xval,
                                                        scoring=scoring
                                                        , return_estimator=True)
            unfair_score = []

            for id_unfair in range(len(unfairness_metrics.UNFAIRNESS_METRICS)):
                if id_unfair == self.unfair_index:
                    unfair_score.append(result['test_'+unfairness_metrics.UNFAIRNESS_METRICS[id_unfair]].mean())
            unfair_score.append(result['test_auc'].mean())
            unfair_score.append(result['test_acc'].mean())
            if pnt:
                print(unfair_score)
            unfair_scores.append(unfair_score[0])
            score.append([sml, unfair_score[0]])

        n_top_10 = int(len(unfair_scores) * 0.1)

        # adjust weight
        if sum(unfair_scores[:n_top_10])/n_top_10 < expect_score:
            if gen_percent > 1 - (expect_score - sum(unfair_scores[:n_top_10])/n_top_10) / (expect_score - self.inital_lr):
                current_lr = sum(unfair_scores[:n_top_10]) / n_top_10
                current_percent = (current_lr - self.inital_lr)/(expect_score - self.inital_lr)
                self.coeff = gen_percent - current_percent + 1

        for i_scores in range(len(score)):
            score[i_scores].insert(0, self.coeff*4*score[i_scores][1] + score[i_scores][0])

        return score, self.coeff

    def post_evaluate(self, population, labels, group, clf):
        if self.groupkfold is not None:
            xval = model_selection.GroupKFold(4)
        else:
            xval = model_selection.KFold(4, shuffle=True, random_state=11798)

        groups_syn = group
        scoring = {}
        for m in unfairness_metrics.UNFAIRNESS_METRICS:
            if m == "calibration":
                metric = unfairness_metrics.UnfairnessMetric(pd.Series(groups_syn), m)
                scoring[m] = metrics.make_scorer(metric, needs_proba=True)
            else:
                metric = unfairness_metrics.UnfairnessMetric(pd.Series(groups_syn), m)
                scoring[m] = metrics.make_scorer(metric)
        scoring['auc'] = metrics.make_scorer(metrics.roc_auc_score)
        scoring['acc'] = metrics.make_scorer(metrics.accuracy_score)

        pipe = pipeline.Pipeline([
            ('standardize', preprocessing.StandardScaler()),
            ('model', clf),
        ])
        if self.groupkfold is not None:
            result = model_selection.cross_validate(pipe, population, pd.Series(labels), verbose=0,
                                                    cv=xval, groups=self.groupkfold,
                                                    scoring=scoring
                                                    , return_estimator=True)
        else:
            result = model_selection.cross_validate(pipe, population, pd.Series(labels), verbose=0,
                                                    cv=xval,
                                                    scoring=scoring
                                                    , return_estimator=True)

        unfair_score = []
        unfair_score.append(result['test_' + unfairness_metrics.UNFAIRNESS_METRICS[self.unfair_index]].mean())
        unfair_score.append(result['test_auc'].mean())
        unfair_score.append(result['test_acc'].mean())
        self.last_unfair = unfair_score[0]

        c = 0
        scores = []
        if self.groupkfold is not None:
            for train_index, test_index in xval.split(self.data, groups=self.groupkfold):
                # X_test, y_test = self.data[test_index], self.labels[test_index]
                X_test, y_test = self.data.iloc[test_index], self.labels.iloc[test_index]
                pred = result['estimator'][c].predict(X_test)
                scores.append(metrics.roc_auc_score(y_test, pred))
        else:
            for train_index, test_index in xval.split(self.data):
                # X_test, y_test = self.data[test_index], self.labels[test_index]
                X_test, y_test = self.data.loc[test_index], self.labels.loc[test_index]
                pred = result['estimator'][c].predict(X_test)
                scores.append(metrics.roc_auc_score(y_test, pred))
                c += 1
        self.score = sum(scores)/len(scores)
        print('AUC ON ORIGINAL DATASET: ', sum(scores)/len(scores))
        return unfair_score[0]


