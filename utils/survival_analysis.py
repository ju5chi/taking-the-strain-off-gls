from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import lifelines as ll
from SurvivalEVAL.Evaluator import LifelinesEvaluator


def _feature_preprocessing_survival_analysis(features: dict) -> pd.DataFrame:
    '''
        feature preprocessing function, that aranges the raw feature data in the 
        right format for the survival analysis models
       ----
        features:
            dictionary containing features
       ----
        Returns:
            the features aranged in a pd.DataFrame
    '''
    features_df = {}
    index = []

    set_index = True
    for f_name, f_values in features.items():
        features_df[f_name] = []
        for group in features[f_name].keys():
            for patient in features[f_name][group].keys():
                if set_index:
                    index.append(patient)

                features_patient = features[f_name][group][patient]

                if type(features_patient) == tuple:
                    if f_name + ' (1Hot)' not in features_df.keys():
                        features_df[f_name + ' (1Hot)'] = []
                    features_df[f_name + ' (1Hot)'].append(features_patient[0])
                    features_df[f_name].append(features_patient[1])
                
                else:
                    features_df[f_name].append(features_patient)
        
        set_index = False
    
    return pd.DataFrame(data=features_df, index=index)


class AbstractSurvivalAnalysis(ABC):
    model_type = 'survival analysis'
    
    def __init__(self):
        '''
            abstract base class for survival analysis models
        '''

        self.model = None
        self.metrics = None
        self.metrics_predict = None

        self.training_times = None
        self.training_censors = None

        self.mean_fit = None
        self.std_fit = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, duration_event_col: tuple[str, str]):
        pass

    def predict(self, X: pd.DataFrame, duration_event_col: tuple[str, str]):
        self.metrics_predict = self._analysis(X, duration_event_col)

    def _preprocess_data(self, X: pd.DataFrame, duration_event_col: tuple[str, str], normalise: bool):
        if normalise:
            mask_explanatory = ~np.isin(X.columns, [duration_event_col[0], duration_event_col[1]])
            self.mean_fit = np.mean(X, axis=0) * mask_explanatory
            self.std_fit = np.std(X, ddof=1, axis=0) * mask_explanatory + ~mask_explanatory * 1.0
        else:
            self.mean_fit = np.zeros(len(X.columns))
            self.std_fit = np.ones(len(X.columns))

        self.training_times = X[duration_event_col[0]]
        self.training_censors = X[duration_event_col[1]]

        normed_X = (X - self.mean_fit) / self.std_fit

        return normed_X

    def _analysis(self, X: pd.DataFrame, duration_event_col: tuple[str, str]):
        metrics = {}

        assert self.mean_fit is not None and self.std_fit is not None, 'Model needs to be fitted first.'
        normed_X = (X - self.mean_fit) / self.std_fit

        metrics['log likelihood'] = self.model.score(normed_X, scoring_method='log_likelihood')
        metrics['concordance index (lifelines)'] = self.model.score(normed_X, scoring_method='concordance_index')

        times = normed_X[duration_event_col[0]]
        surv_times = pd.concat([pd.Series([0.0]), times]) if times.iloc[0] > 0 else times
        censoring_ind = normed_X[duration_event_col[1]]

        surv_function = self.model.predict_survival_function(normed_X, times=surv_times)

        evl = LifelinesEvaluator(surv_function, times, censoring_ind, self.training_times, self.training_censors)

        metrics['concordance index (SurvEVAL)'] = evl.concordance(ties='All')[0]
        metrics['IBS'] = evl.integrated_brier_score()
        metrics['D-calibration'] = evl.d_calibration()[0]
        metrics['1-calibration (1y)'] = evl.one_calibration(365)[0]
        metrics['1-calibration (2.5y)'] = evl.one_calibration(2.5*365)[0]
        metrics['1-calibration (5y)'] = evl.one_calibration(5*365)[0]

        return metrics


class AbstractRegSurvivalAnalysis(AbstractSurvivalAnalysis):

    def __init__(self):
        '''
            abstract base class for regularised survival analysis models
        '''
        super().__init__()

    @staticmethod
    @abstractmethod
    def complexity(lams_dict: dict):
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, duration_event_col: tuple[str, str], lams_dict: dict, normalise: bool):
        pass


class CoxProportionalHazardsModel(AbstractSurvivalAnalysis):
    
    def __init__(self):
        '''
            implements the Cox proportional hazards model
        '''
        super().__init__()

    def fit(self, X: pd.DataFrame, duration_event_col: tuple[str, str], normalise: bool = True):
        normed_X = self._preprocess_data(X, duration_event_col, normalise)
        cph = ll.CoxPHFitter()

        cph.fit(normed_X, duration_event_col[0], duration_event_col[1])
        
        self.model = cph
        self.coeffs = cph.params_.to_numpy().reshape(1, -1)
        self.metrics = self._analysis(X, duration_event_col)


class L1RegCoxProportionalHazardsModel(AbstractRegSurvivalAnalysis):

    def __init__(self):
        '''
            implements a l1-regularised version of the Cox proportional hazards model
        '''
        super().__init__()

    @staticmethod
    def complexity(lams_dict: dict):
        return 1/lams_dict['alpha']

    def fit(self, X: pd.DataFrame, duration_event_col: tuple[str, str], lams_dict: dict, normalise: bool = True):
        normed_X = self._preprocess_data(X, duration_event_col, normalise)
        cph = ll.CoxPHFitter(penalizer=lams_dict['alpha'], l1_ratio=1.0)

        cph.fit(normed_X, duration_event_col[0], duration_event_col[1])
        
        self.model = cph
        self.coeffs = cph.params_.to_numpy().reshape(1, -1)
        self.metrics = self._analysis(X, duration_event_col)


class L2RegCoxProportionalHazardsModel(AbstractRegSurvivalAnalysis):

    def __init__(self):
        '''
            implements a l2-regularised version of the Cox proportional hazards model
        '''
        super().__init__()
    
    @staticmethod
    def complexity(lams_dict: dict):
        return 1/lams_dict['alpha']

    def fit(self, X: pd.DataFrame, duration_event_col: tuple[str, str], lams_dict: dict, normalise: bool = True):
        normed_X = self._preprocess_data(X, duration_event_col, normalise)
        cph = ll.CoxPHFitter(penalizer=lams_dict['alpha'], l1_ratio=0.0)

        cph.fit(normed_X, duration_event_col[0], duration_event_col[1])
        
        self.model = cph
        self.coeffs = cph.params_.to_numpy().reshape(1, -1)
        self.metrics = self._analysis(X, duration_event_col)
