from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import statsmodels.api as sm

from sklearn.metrics import roc_curve, auc

from .constants import PREC


def _feature_preprocessing_classification(features: dict, 
                                          response: dict
                                          ) -> tuple[pd.DataFrame, np.ndarray]:
    '''
        feature preprocessing function, that aranges the raw feature data in the 
        right format for the classification models
       ----
        features:
            dictionary containing features
        response:
            the response variable, i.e. the classes
       ----
        Returns:
            the features aranged in a pd.DataFrame and the response separately in 
            a np.ndarray
    '''
    features_df = {}
    index = []
    classes_arr = []

    set_index_classes = True
    for f_name, f_values in features.items():
        features_df[f_name] = []
        for group in features[f_name].keys():
            for patient in features[f_name][group].keys():
                if set_index_classes:
                    index.append(patient)
                    classes_arr.append(response[group][patient])

                features_df[f_name].append(features[f_name][group][patient])
        
        set_index_classes = False
    
    return pd.DataFrame(data=features_df, index=index), np.array(classes_arr)


class AbstractLinearModel(ABC):
    model_type = 'classification'

    classes_to_int = {'control': 0, 'disease': 1}
    int_to_classes = {0: 'control', 1: 'disease'}

    def __init__(self):
        '''
            abstract base class for classifiers that can be reduced to a linear 
            decision rule
        '''

        self.scores = None
        self.indices = None
        self.classes = None

        self.scores_predict = None
        self.indices_predict = None
        self.classes_predict = None

        self.intercept = None
        self.coeffs = None

        self.optimal_threshold = None

        self.mean_fit = None
        self.std_fit = None

        self.metrics = None
        self.metrics_predict = None

        self.roc_curve_values = None
        self.roc_curve_values_predict = None

        self.thresholds = None
        self.thresholds_predict = None

    @abstractmethod
    def fit(self, scores_methods: dict):
        pass

    def _exchange_labels_for_int(self, arr: np.ndarray):
        arr_int = np.zeros(arr.shape)
        for label, i in self.classes_to_int.items():
            mask = arr == label
            arr_int[mask] = i
        
        return arr_int

    def _preprocess_data(self, X: pd.DataFrame, y: np.ndarray, normalise: bool):
        self.classes = y
        self.indices = X.index.to_numpy()
        X = X.to_numpy()

        if normalise:
            self.mean_fit = np.mean(X, axis=0)
            self.std_fit = np.std(X, ddof=1, axis=0)
        else:
            self.mean_fit = np.zeros(X.shape[-1])
            self.std_fit = np.ones(X.shape[-1])

        normed_X = (X - self.mean_fit) / self.std_fit

        return normed_X, self._exchange_labels_for_int(y)

    def predict(self, X: pd.DataFrame, y: np.ndarray, **kwargs):
        self.classes_predict = y
        self.indices_predict = X.index.to_numpy()
        X = X.to_numpy()

        assert self.mean_fit is not None and self.std_fit is not None, 'Model needs to be fitted first.'
        assert self.coeffs is not None and self.intercept is not None, 'Model needs to be fitted first.'

        normed_X = (X - self.mean_fit) / self.std_fit
        self.scores_predict = (self.intercept + normed_X @ self.coeffs[0]).reshape(-1,)

        self.metrics_predict, self.roc_curve_values_predict, self.thresholds_predict \
            = self._roc_analysis(self.scores_predict, self.classes_predict, self.indices_predict, **kwargs)
    
    def _roc_analysis(self, scores, classes, patients, dpc, use_intercept):
        if scores is None:
            raise RuntimeError('Need to fit the model before calling `roc_analyisis`.')

        roc_scores = {}

        mast_type = []
        type_to_num = {}
        if dpc is not None:
            set_of_types = set([dpc[group][patient] for group in dpc.keys() for patient in dpc[group].keys()])
            # assign increasing integers to the labels (after sorting them in alphabetical order)
            type_to_num = {t: i for i, t in enumerate(sorted(set_of_types))}

            for group, patient in zip(classes, patients):
                mast_type.append(type_to_num[dpc[group][patient]])
            mast_type = np.array(mast_type)

        # calculate the ROC curve using the implementation from scikit-learn
        fpr, tpr, thresholds = roc_curve(y_true=classes, y_score=scores, pos_label='disease')
        roc_curve_values = [[fpr[i], tpr[i]] for i in range(len(fpr))]

        # by default, use the intercept found by the model (i.e. 0 is the decision boundary)
        self.optimal_threshold = 0.0

        # otherwise, choose the threshold value found by some criterion involving the ROC curve
        if not use_intercept:
            opt_idx = self._select_opt_roc_idx([fpr, tpr])
            self.optimal_threshold = thresholds[opt_idx]

        c_matrix = self._compute_confusion_matrix(classes, scores, self.optimal_threshold)

        metrics = {}
        metrics[r'$ACC(\theta)$'] = self._acc(c_matrix)
        metrics[r'$ACC_b(\theta)$'] = self._weighted_acc(c_matrix)
        metrics[r'$Spec(\theta)$'] = self._specificity(c_matrix)
        metrics[r'$Sens(\theta)$'] = self._sensitivity(c_matrix)
        metrics[r'$AUC$'] = self._auc([fpr, tpr])
        
        # compute again some metrics for threshold = 0:
        c_matrix_0 = self._compute_confusion_matrix(classes, scores, 0.0)
        metrics[r'$ACC(0)$'] = self._acc(c_matrix_0)
        metrics[r'$ACC_b(0)$'] = self._weighted_acc(c_matrix_0)
        metrics[r'$Spec(0)$'] = self._specificity(c_matrix_0)
        metrics[r'$Sens(0)$'] = self._sensitivity(c_matrix_0)

        if dpc is not None:
            metrics[r'$\rho_{n; X, Y_{I_c}}$ (all)'] = self._pearson_corr(scores, mast_type)

            # compute the mast type correlation with the score in the disease group
            a = scores[classes == 'disease']
            b = mast_type[classes == 'disease']
            metrics[r'$\rho_{n; X, Y_{I_c}}$ (diseased)'] = self._pearson_corr(a, b)
            metrics['corr spearman (diseased)'] = self._spearman_rank_corr(a, b)
            metrics['corr kendall tau b (diseased)'] = self._kendall_tau_b_corr(a, b)
            
        return metrics, roc_curve_values, thresholds
    
    @staticmethod
    def _select_opt_roc_idx(roc):
        fpr, tpr = roc
        # select the index i, such that (fpr[i], tpr[i]) maximises the affine 
        # linear objective function ((-1, 0) + (fpr[i], tpr[i])) @ (-1, 1)
        objective = (tpr - (fpr + 1)) / 2**0.5
        opt_idx = np.argmax(objective) 
        return opt_idx

    @staticmethod
    def _auc(roc):
        return auc(roc[0], roc[1])
    
    @staticmethod
    def _compute_confusion_matrix(y_true, y_score, opt_threshold):
        mask_disease = y_score >= opt_threshold
        y_pred = np.full(y_score.shape, 'control')
        y_pred[mask_disease] = 'disease'

        # build the confusion matrix with true classes along rows and predicted 
        # classes along the columns
        c_matrix = np.zeros((2, 2))
        labels = ['disease', 'control']
        for i, t_label in enumerate(labels):
            for j, p_label in enumerate(labels):
                c_matrix[i, j] = np.sum((y_true == t_label) & (y_pred == p_label))

        assert abs(np.sum(c_matrix) - len(y_true)) < PREC, 'Confusion matrix contains double countings.'
        
        return c_matrix

    @staticmethod
    def _acc(c_matrix):
        return (c_matrix[0, 0] + c_matrix[1, 1]) / np.sum(c_matrix)

    def _weighted_acc(self, c_matrix):
        return 0.5 * (self._specificity(c_matrix) + self._sensitivity(c_matrix))
    
    @staticmethod
    def _specificity(c_matrix):
        # specificity = TNR = TN / N 
        return c_matrix[1, 1] / (np.sum(c_matrix[1, :]))

    @staticmethod
    def _sensitivity(c_matrix):
        # sensitivity = TPR = TP / P
        return c_matrix[0, 0] / (np.sum(c_matrix[0, :]))

    @staticmethod
    def _pearson_corr(x, y):
        assert len(x.shape) <= 1 and len(y.shape) <= 1
        return stats.pearsonr(x, y).statistic

    @staticmethod
    def _spearman_rank_corr(x, y):
        assert len(x.shape) <= 1 and len(y.shape) <= 1
        return stats.spearmanr(x, y).statistic

    @staticmethod
    def _kendall_tau_b_corr(x, y):
        assert len(x.shape) <= 1 and len(y.shape) <= 1
        return stats.kendalltau(x, y, variant='b').statistic


class AbstractRegLinearModel(AbstractLinearModel):

    def __init__(self):
        '''
            abstract base class for classifiers that can be reduced to a linear 
            decision rule and that are trained using some built in regularisation
            functionality
        '''
        super().__init__()

    @staticmethod
    @abstractmethod
    def complexity(lams_dict: dict):
        pass

    @abstractmethod
    def fit(self, scores_methods: dict, lams_dict: dict):
        pass


class NoModel(AbstractLinearModel):

    def __init__(self):
        '''
            linear classifier that has intercept 0.0 and coefficients for the
            explanatory variables that are all 1.0
        '''
        super().__init__()

    def fit(self, X: pd.DataFrame, y: np.ndarray, normalise=True, **kwargs):   
        normed_X, y = self._preprocess_data(X, y, normalise)
        coef = np.ones((1, normed_X.shape[1]))
        # now produce the combined scores from the underlying linear model
        self.scores = (normed_X @ coef).reshape(-1,)
        self.intercept = np.zeros((1, 1))
        self.coeffs = coef

        self.metrics, self.roc_curve_values, self.thresholds\
            = self._roc_analysis(self.scores, self.classes, self.indices, **kwargs)


class LogisticRegressionModel(AbstractLinearModel):

    def __init__(self):
        '''
            linear classifier that uses logistic regression to find the model
            coefficients
        '''
        super().__init__()

    def fit(self, X: pd.DataFrame, y: np.ndarray, normalise=True, **kwargs):
        normed_X, y = self._preprocess_data(X, y, normalise)

        design = sm.add_constant(normed_X)
        lr = sm.GLM(y, design, family=sm.families.Binomial(link=sm.families.links.Logit())).fit()

        # now produce the combined scores from the underlying linear model
        self.scores = lr.params[0] + normed_X @ lr.params[1:]

        # store the coefficients found by logistic regression
        self.intercept = lr.params[0].reshape(1, -1) 
        self.coeffs = lr.params[1:].reshape(1, -1) 

        self.metrics, self.roc_curve_values, self.thresholds\
            = self._roc_analysis(self.scores, self.classes, self.indices, **kwargs)


class L1RegLogisticRegressionModel(AbstractRegLinearModel):

    def __init__(self):
        '''
            linear classifier that uses l1-regularised logistic regression to 
            find the model coefficients
        '''
        super().__init__()

    @staticmethod
    def complexity(lams_dict):
        return 1/lams_dict['alpha']

    def fit(self, X: pd.DataFrame, y: np.ndarray, lams_dict: dict, normalise=True, **kwargs):
        normed_X, y = self._preprocess_data(X, y, normalise)

        design = sm.add_constant(normed_X)
        lr = sm.GLM(y, 
                    design, 
                    family=sm.families.Binomial(link=sm.families.links.Logit())
                    ).fit_regularized(**lams_dict, L1_wt=1.0)

        # now produce the combined scores from the underlying linear model
        self.scores = lr.params[0] + normed_X @ lr.params[1:]

        # store the coefficients found by logistic regression
        self.intercept = lr.params[0].reshape(1, -1) 
        self.coeffs = lr.params[1:].reshape(1, -1) 

        self.metrics, self.roc_curve_values, self.thresholds\
            = self._roc_analysis(self.scores, self.classes, self.indices, **kwargs)


class L2RegLogisticRegressionModel(AbstractRegLinearModel):

    def __init__(self):
        '''
            linear classifier that uses l2-regularised logistic regression to 
            find the model coefficients
        '''
        super().__init__()

    @staticmethod
    def complexity(lams_dict):
        return 1/lams_dict['alpha']

    def fit(self, X: pd.DataFrame, y: np.ndarray, lams_dict: dict, normalise=True, **kwargs):
        normed_X, y = self._preprocess_data(X, y, normalise)

        design = sm.add_constant(normed_X)
        lr = sm.GLM(y, 
                    design, 
                    family=sm.families.Binomial(link=sm.families.links.Logit())
                    ).fit_regularized(**lams_dict, L1_wt=0.0)

        # now produce the combined scores from the underlying linear model
        self.scores = lr.params[0] + normed_X @ lr.params[1:]

        # store the coefficients found by logistic regression
        self.intercept = lr.params[0].reshape(1, -1) 
        self.coeffs = lr.params[1:].reshape(1, -1) 

        self.metrics, self.roc_curve_values, self.thresholds\
            = self._roc_analysis(self.scores, self.classes, self.indices, **kwargs)


class LinearDiscriminantModel(AbstractLinearModel):

    def __init__(self):
        '''
            linear classifier that uses linear discriminant analysis to find the 
            model coefficients
        '''
        super().__init__()

    def fit(self, X: pd.DataFrame, y: np.ndarray, normalise=True, **kwargs):
        normed_X, y = self._preprocess_data(X, y, normalise)

        lda = LinearDiscriminantAnalysis()
        lda.fit(normed_X, y)

        # now produce the combined scores from the underlying linear model
        self.scores = lda.intercept_ + normed_X @ lda.coef_[0]

        # store the coefficients found by LDA
        self.intercept = lda.intercept_
        self.coeffs = lda.coef_

        self.metrics, self.roc_curve_values, self.thresholds\
            = self._roc_analysis(self.scores, self.classes, self.indices, **kwargs)
