import os
import typing as t
from itertools import combinations, product
import random
import copy
import math
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .constants import PATH_TO_RESULTS, PREC
from .data import EchocardiographyDataCollection
from .tools import NumpyEncoder
from .evaluation import plot_scores

from .survival_analysis import _feature_preprocessing_survival_analysis, AbstractSurvivalAnalysis, AbstractRegSurvivalAnalysis
from .classification import _feature_preprocessing_classification, AbstractLinearModel, AbstractRegLinearModel, NoModel


def _cross_val_split(idxs: list, n_folds: int, seed: int) -> list[list[int]]:
    '''
        function that splits a set of indices into `n_folds`-many (almost) equally
        sized folds
       ----
        idxs:
            a list of indices that are to be partitioned into folds
        n_folds:
            the number of folds 
        seed:
            the seed to be passed to the random number generator
       ----
        Returns:
            a list of lists that contain the indices associated to the folds;
            if n_folds <= 1, then the function returns [[]].
    '''
    assert len(set(idxs)) == len(idxs), 'There are duplicate indices.'

    # early return if there is only one fold (no individuals are in the 
    # validation set)
    if n_folds <= 1:
        return [[]]

    idxs_copy = copy.deepcopy(idxs)
    random.seed(seed)
    random.shuffle(idxs_copy)
    
    size_fold = len(idxs_copy) // n_folds
    folds = [[] for _ in range(n_folds)]
    for i in range(len(idxs_copy)):
        folds[i%n_folds].append(idxs_copy[i])

    assert set.intersection(*[set(f) for f in folds]) == set(), 'The folds are not disjoint.'
    lengths_folds = [len(f) for f in folds]
    assert max(lengths_folds) - min(lengths_folds) < 2, 'The folds are not equally sized.'
    assert set.union(*[set(f) for f in folds]) == set(idxs), 'The folds are not exhaustive in the set of all indices.'

    return folds



class ModelSelection():

    def __init__(self, 
                 features: dict, 
                 response: dict | tuple, 
                 models: dict, 
                 dpc: None | dict = None
                 ) -> None:
        '''
            a class for performing model selection
           ----
            features:
                a dictionary of features
            response:
                either a dictionary containing the classes of the whole sample
                for classification, or a tuple with the indices of the event times
                and the censoring indicator in the case of survival analysis
            models:
                a dictionary containing the model classes to be used; if the model is 
                unregularised, then the dictionary values is the uninstantiated model
                class; if the model is regularised, then the dictionary value conatains
                a tuple 
                    (m, {'hyper-param. 1': [...], 'hyper-param. 1': [...], ...}), 
                where the first entry is the uninstantiated model and the second entry
                contains a dictionary with ranges of hyper-parameter values
            dpc:
                a dictionary containing deformation pattern classes 
        '''

        self.features = None
        self.response = None
        self.dpc = dpc
        self.folds = []

        model_base_to_general_type = {AbstractLinearModel: 'classification', 
                                      AbstractRegLinearModel: 'classification',
                                      AbstractSurvivalAnalysis: 'survival analysis',
                                      AbstractRegSurvivalAnalysis: 'survival analysis'
                                      }

        # check if all specified models are of the same general type
        self.model_type = None
        for entry in models.values():
            
            if type(entry) == tuple:
                general_type = model_base_to_general_type[entry[0].__base__]

                if self.model_type is None:
                    self.model_type = general_type
                    continue

                elif not self.model_type == general_type:
                    raise ValueError('The given model classes are of conflicting types.')

            elif self.model_type is None:
                self.model_type = model_base_to_general_type[entry.__base__]
                continue

            elif self.model_type != model_base_to_general_type[entry.__base__]:
                raise ValueError('The given model classes are of conflicting types.')


        if self.model_type == 'survival analysis':
            self.features = _feature_preprocessing_survival_analysis(features)
            self.response = response

        elif self.model_type == 'classification':
            self.features, self.response = _feature_preprocessing_classification(features, response)

        else:
            raise ValueError(f'The given model type `{self.model_type}` is not supported.')

        self.models = models
        self.results = {}
        self.optimal_models = None


    def save_optimal_models_to_json(self, filename_json: str) -> None:
        output = {'features': self.features,
                  'folds': self.folds,
                  'results': {}
                 }

        for model in self.optimal_models:
            model_name = model['model name']
            output['results'][model_name] = {'features': model['features'],
                                             'results folds': []
                                            }
            for m in model['results folds']:
                if self.model_type == 'classification':
                    output['results'][model_name]['results folds'].append(
                        {'coeffs': m.coeffs,
                        'intercept': m.intercept,
                        'scores predict': m.scores_predict
                        }
                        )

                elif self.model_type == 'survival analysis':
                    raise NotImplementedError

        if not os.path.isdir(PATH_TO_RESULTS):
            os.mkdir(PATH_TO_RESULTS)

        # dump the converted data to json
        with open(PATH_TO_RESULTS + filename_json, 'w') as out_file:
            json.dump(output, out_file, cls=NumpyEncoder)

        
    @staticmethod
    def _calculate_cv_mean_std_err(results_folds: list, 
                                   objective_function: None | t.Callable = None, 
                                   kind: str = 'predict'
                                   ) -> tuple[float, float]:
        '''
            calculation of the CV mean and its standard error for different types 
            of variables
           ----
            results_folds:
                the list of models that is generated during cross-validation
            objective_function:
                some scalar value of the model that should be estimated; also
                depends on the next variable
            kind:
                the general aspect of the model that is to be analysed; can be any
                of "threshold", "intercept", "coefficients", or "predict"
           ----
            Returns:
                mean:
                    the CV mean
                std_err:
                    the CV standard error for the above mean estimate
        '''

        cv_objective_values = None

        if kind == 'threshold':
            cv_objective_values = np.array([d.optimal_threshold for d in results_folds])
            # avoid silly behaviour, when one of the thresholds is inf or -inf
            mask_infs = np.isinf(cv_objective_values) 
            cv_objective_values = cv_objective_values[~mask_infs]

        elif kind == 'intercept':
            cv_objective_values = np.array([d.intercept for d in results_folds])

        elif kind == 'coefficients':
            if objective_function is None:
                raise ValueError(f'objective_function needs to be specified, when calling kind `{kind}`.')
            cv_objective_values = np.array([objective_function(d.coeffs) for d in results_folds])

        elif kind == 'predict':
            if objective_function is None:
                raise ValueError(f'objective_function needs to be specified, when calling kind `{kind}`.')
            cv_objective_values = np.array([objective_function(d.metrics_predict) for d in results_folds])

        else:
            raise ValueError(f'Kind `{kind}` is not a valid value.')

        # CV estimates for the mean and the standard error
        # the CV fold sizes are almost equal (sizes differ at most by 1)
        n_folds = len(cv_objective_values)
        mean = np.mean(cv_objective_values)
        std_err = (np.var(cv_objective_values, ddof=1) / n_folds)**0.5

        return mean, std_err

    def _features_train_val_split(self, 
                                  val_indices: list[int], 
                                  selected_feature_names: list[str] 
                                  ) -> tuple:
        '''
            splits the features into a training and a validation set
           ----
            val_indices:
                list of row-indices of the feature dataframe that correspond to 
                the individuals in the validation dataset
            selected_feature_names:
                list of a subset of the features that should be extracted
           ----
            Returns:
                if the studied model is a survival analysis model, then a tuple
                (f_train, r, f_val, r) is returned, where f_train & f_val are the
                training- & validation features, and r denotes the response;
                if the studied model is a classification model, then a tuple
                (f_train, r_train, f_val, r_val) is returned, where f_train & f_val 
                are the training- & validation features, and r_train & r_val are 
                the training- & validation responses
        '''
        if self.model_type == 'survival analysis':
            selected_feature_names = (*selected_feature_names, self.response[0], self.response[1])

        # use of val_indices is important, to keep the same order sepcified in val_indices (for saving the values)
        features_val = self.features.loc[val_indices, selected_feature_names] 
        # for the training data of the fold, this information does not exist
        mask_train = ~self.features.index.isin(val_indices)
        features_train = self.features.loc[mask_train, selected_feature_names]

        if self.model_type == 'survival analysis':
            return features_train, self.response, features_val, self.response

        elif self.model_type == 'classification':
            # same care has to be taken for the class labels
            classes_indices = np.array([self.features.index.get_loc(v) for v in val_indices])

            return features_train, self.response[mask_train], features_val, self.response[classes_indices]

    def _evaluate_unregularised_model(self, 
                                      model, 
                                      model_name: str, 
                                      kind: str, 
                                      objective_function: t.Callable, 
                                      n_folds: int, 
                                      seed: int, 
                                      verbose: bool, 
                                      use_intercept: bool, 
                                      normalise: bool, 
                                      maximum: bool
                                      ) -> None:        
        '''
            performs the model selection procedure for unregularised models
           ----
            model:
                the uninstantiated model class, can be either a child class of 
                AbstractLinearModel or AbstractSurvivalAnalysis
            model_name:
                the name of the model
            kind:
                what kind of feature selection procedure should be applied; can
                be either one of "exhaustive" (for an exhaustive search in the
                space of all subsets of features) or "forward" (for a forward
                selection procedure); in the latter case, the objective function
                needs to be specified
            objective_function:
                needed for forward selection of the features to assess which 
                features should be chosen; gets passed to the function 
                `_calculate_cv_mean_std_err`
            n_folds:
                the number of cross-validation folds
            seed:
                the seed for the random number generator
            verbose:
                whether there should be some console output about the progress
            use_intercept:
                whether the linear models should use the intercept instead of a
                subsequently chosen threshold value
            normalise:
                whether the data is to be normalised on each fold of cross-
                validation
            maximum:
                whether the objective function needs to be maximised or minimised
        '''

        feature_names = list(self.features.columns)
        
        if self.model_type == 'survival analysis':
            temp_feature_names = []
            for feature in feature_names:
                if feature not in self.response:
                    temp_feature_names.append(feature)
            feature_names = temp_feature_names
        
        # group 1-hot encoded features
        temp_feature_names = []
        for f_name in feature_names:
            if f_name + ' (1Hot)' in feature_names:
                temp_feature_names.append((f_name, f_name + ' (1Hot)'))
            elif '(1Hot)' not in f_name:
                temp_feature_names.append((f_name,))
        feature_names = temp_feature_names
                
        max_n_features = 0
        for f_name_tup in feature_names:
            max_n_features += len(f_name_tup)

        # special case: when there is only one feature available, use the mode NoModel
        if max_n_features < 2 and self.model_type == 'classification':
            model = NoModel

        self.results[model_name] = []
        self.folds = _cross_val_split(idxs=list(self.features.index), n_folds=n_folds, seed=seed)

        chosen_feature_combs = []

        if kind == 'exhaustive':
            n_models = sum([math.comb(len(feature_names), k) for k in range(1, len(feature_names)+1)])

        elif kind == 'forward':
            n_models = len(feature_names)
            if objective_function is None:
                raise ValueError(f'If `kind` == `forward`, then `objective_function` needs to be not None.')
        
        else:
            raise ValueError(f'Kind `{kind}` is not a valid value.')

        count_models = 0
        n_samples = len(self.features.index)
        n_features = 1

        while n_features <= max_n_features:
            
            iter_feature_combinations = None
            n_feature_components = None

            if kind == 'exhaustive':
                iter_feature_combinations = combinations(feature_names, n_features)

            elif kind == 'forward':
                iter_feature_combinations = []
                n_feature_components = []
                # append all remaining combinations of `n_features` many 
                # features to the search list
                if len(chosen_feature_combs) > 0:
                    for tup in chosen_feature_combs:
                        if len(tup) != n_features-1:
                            continue

                        for feature in feature_names:
                            if len(feature) == 1 and feature[0] not in tup:
                                iter_feature_combinations.append((*tup, feature[0]))
                                n_feature_components.append(len(feature))

                            if len(feature) == 2 and feature[0] not in tup and feature[1] not in tup:
                                iter_feature_combinations.append((*tup, *feature))
                                n_feature_components.append(len(feature))

                else:
                    for feature in feature_names:
                        iter_feature_combinations.append(feature)
                        n_feature_components.append(len(feature))
        
                forward_search_candidate = (None, None, None)
               

            chosen_feature_components = 0

            for comb, n_components in zip(iter_feature_combinations, n_feature_components):
                results_folds = []

                for fold in self.folds:
                    m = model()
                    
                    t_X, t_y, v_X, v_y = self._features_train_val_split(val_indices=fold, selected_feature_names=comb)

                    if self.model_type == 'classification':
                        m.fit(t_X, t_y, normalise=normalise, dpc=self.dpc, use_intercept=use_intercept)
                    elif self.model_type == 'survival analysis':
                        m.fit(t_X, t_y, normalise=normalise)

                    fold_size_predict = len(fold)
                    if len(fold) == 0:
                        v_X, v_y = t_X, t_y
                        fold_size_predict = n_samples

                    if self.model_type == 'classification':
                        m.predict(v_X, v_y, dpc=self.dpc, use_intercept=use_intercept)
                    elif self.model_type == 'survival analysis':
                        m.predict(v_X, v_y)

                    results_folds.append(m)

                if kind == 'forward':
                    cv_mean, _ = self._calculate_cv_mean_std_err(results_folds, objective_function)
                    sign = 1 if maximum else -1
                    if forward_search_candidate[0] is None or sign*cv_mean > sign*forward_search_candidate[0]:
                        forward_search_candidate = (cv_mean, results_folds, comb)
                        chosen_feature_components = n_components

                if kind == 'exhaustive':
                    self.results[model_name].append({'results folds': results_folds,
                                                     'model name': model_name,
                                                     'features': comb
                                                    }
                                                   )
                
                    count_models += 1

                    if verbose:
                        perc = count_models / n_models * 100
                        n_whitespaces = len(str(n_models)) - len(str(count_models))
                        print(f'exhaustive_evaluation: ' + ' '*n_whitespaces + f'{count_models}/{n_models} ({perc:.2f}%) done.')
            
            if kind == 'forward':
                self.results[model_name].append({'results folds': forward_search_candidate[1],
                                                 'model name': model_name,
                                                 'features': forward_search_candidate[2]
                                                }
                                               )

                chosen_feature_combs.append(forward_search_candidate[2])
                count_models += 1

                if verbose:
                    perc = count_models / n_models * 100
                    n_whitespaces = len(str(n_models)) - len(str(count_models))
                    print(f'forward_evaluation: ' + ' '*n_whitespaces + f'{count_models}/{n_models} ({perc:.2f}%) done.')

            n_features += chosen_feature_components

    
    def _evaluate_regularised_model(self, 
                                    model, 
                                    model_name: str, 
                                    tuning_parameters: dict, 
                                    n_folds: int, 
                                    seed: int, 
                                    verbose: bool, 
                                    use_intercept: bool, 
                                    normalise: bool
                                    ) -> None:
        '''
            performs the model selection procedure for regularised models
           ----
            model:
                the uninstantiated model class, can be either a child class of 
                AbstractLinearModel or AbstractSurvivalAnalysis
            model_name:
                the name of the model
            tuning_parameters:
                a dictionary containing ranges of tuning-parameter values whose
                performance needs to be estimated by cross-validation
            n_folds:
                the number of cross-validation folds
            seed:
                the seed for the random number generator
            verbose:
                whether there should be some console output about the progress
            use_intercept:
                whether the linear models should use the intercept instead of a
                subsequently chosen threshold value
            normalise:
                whether the data is to be normalised on each fold of cross-
                validation
        '''
        feature_names = list(self.features.columns)
        if self.model_type == 'survival analysis':
            temp_feature_names = []
            for feature in feature_names:
                if feature not in self.response:
                    temp_feature_names.append(feature)
            feature_names = temp_feature_names
        
        # special case: when there is only one feature available, use the mode NoModel
        if len(feature_names) < 2 and self.model_type == 'classification':
            model = NoModel

        self.results[model_name] = []
        self.folds = _cross_val_split(idxs=list(self.features.index), n_folds=n_folds, seed=seed)

        n_models = np.prod([len(t_params) for t_params in tuning_parameters.values()])

        count_models = 0
        n_samples = len(self.features.index)

        feature_selections_train = []
        feature_selections_val = []

        # the list of selected features to fit the model stays the same, so it is 
        # possible to store the feature scores split up in the folds in advance
        for fold in self.folds:
            t_X, t_y, v_X, v_y = self._features_train_val_split(val_indices=fold, selected_feature_names=feature_names)
            feature_selections_train.append((t_X, t_y))
            feature_selections_val.append((v_X, v_y))

        for lams in product(*tuning_parameters.values()):
            results_folds = []
            lams_dict = {k: lam for k, lam in zip(tuning_parameters.keys(), lams)}
            chosen_feature_combs = []
            complexity = None

            for i, fold in enumerate(self.folds):
                m = model()

                t_X, t_y = feature_selections_train[i]

                if self.model_type == 'classification':
                    m.fit(t_X, t_y, lams_dict=lams_dict, normalise=normalise, dpc=self.dpc, use_intercept=use_intercept)
                elif self.model_type == 'survival analysis':
                    m.fit(t_X, t_y, lams_dict=lams_dict, normalise=normalise)

                v_X, v_y = t_X, t_y
                fold_size_predict = n_samples

                if len(fold) > 0:
                    v_X, v_y = feature_selections_val[i]
                    fold_size_predict = len(fold)

                if self.model_type == 'classification':
                    m.predict(v_X, v_y, dpc=self.dpc, use_intercept=use_intercept)
                elif self.model_type == 'survival analysis':
                    m.predict(v_X, v_y)

                results_folds.append(m)

                # the features whose coefficients are not numerically zero are 
                # considered to be chosen by the model the chosen features accumulated 
                # across folds, i.e. features that are > 0 at least once are chosen
                chosen_feature_combs += [t_X.columns[j] for j, coeff in enumerate(m.coeffs[0]) if np.abs(coeff) > PREC]

                if complexity is None:
                    complexity = m.complexity(lams_dict)

            self.results[model_name].append({'results folds': results_folds,
                                             'model name': model_name,
                                             'features': list(set(chosen_feature_combs)),
                                             'complexity': complexity
                                            }
                                           )

            count_models += 1

            if verbose:
                perc = count_models / n_models * 100
                n_whitespaces = len(str(n_models)) - len(str(count_models))
                print(f'regularised_evaluation: ' + ' '*n_whitespaces + f'{count_models}/{n_models} ({perc:.2f}%) done.')


    def evaluation(self, 
                   kind='forward', 
                   objective_function=None, 
                   n_folds=1, 
                   seed=0, 
                   verbose=False, 
                   normalise=True, 
                   maximum=True
                   ) -> None:
        '''
            a wrapper function for the two methods `_evaluate_unregularised_model`
            and `_evaluate_regularised_model` that chooses the right one depending 
            on the input
           ----
            kind:
                what kind of feature selection procedure should be applied; can
                be either one of "exhaustive" (for an exhaustive search in the
                space of all subsets of features) or "forward" (for a forward
                selection procedure); in the latter case, the objective function
                needs to be specified
            objective_function:
                needed for forward selection of the features to assess which 
                features should be chosen; gets passed to the function 
                `_calculate_cv_mean_std_err`
            n_folds:
                the number of cross-validation folds
            seed:
                the seed for the random number generator
            verbose:
                whether there should be some console output about the progress
            normalise:
                whether the data is to be normalised on each fold of cross-
                validation
            maximum:
                whether the objective function needs to be maximised or minimised
        '''
        for model_name, entry in self.models.items():
            # handle the case where the model is a model with additional parameters;
            # then call the appropriate evaluation function depending on which 
            # abstract base class is the parent
            if type(entry) == tuple:
                if entry[0].__base__ in [AbstractRegLinearModel, AbstractRegSurvivalAnalysis]:
                    self._evaluate_regularised_model(model=entry[0],
                                    model_name=model_name,
                                    tuning_parameters=entry[1], 
                                    n_folds=n_folds, 
                                    seed=seed, 
                                    verbose=verbose,
                                    use_intercept=entry[2] if len(entry) > 2 else False,
                                    normalise=normalise
                                    )
                elif entry[0].__base__ == AbstractLinearModel:
                    self._evaluate_unregularised_model(model=entry[0],
                            model_name=model_name,
                            kind=kind, 
                            objective_function=objective_function, 
                            n_folds=n_folds, 
                            seed=seed, 
                            verbose=verbose,
                            use_intercept=entry[1] if len(entry) > 1 else False,
                            normalise=normalise,
                            maximum=maximum
                            )

            elif entry.__base__ in [AbstractLinearModel, AbstractSurvivalAnalysis]:
                self._evaluate_unregularised_model(model=entry,
                            model_name=model_name,
                            kind=kind, 
                            objective_function=objective_function, 
                            n_folds=n_folds, 
                            seed=seed, 
                            verbose=verbose,
                            use_intercept=False,
                            normalise=normalise,
                            maximum=maximum
                            )
        
            else: 
                raise ValueError('The given model does not have a valid base class.')


    def _find_optimal_model_in_list(self, 
                                    results_list: list[dict], 
                                    objective_function: t.Callable, 
                                    regularisation: None | str | int | float,
                                    maximum: bool = True
                                    ) -> dict:
        '''
            finds the model with the best score in the given list
           ----
            results_list:
                list containing dictionaries with model results as generated by the
                `evaluation` function
            objective_function:
                the objective that is used to rank the models according to their
                performance
            regularisation:
                if not None, it can be either '1 SE' (one-standard-error-
                regularisation) or an integer/float (specifies the maximum level 
                of complexity that should not be exceeded)
            maximum:
                whether the objective function needs to be maximised or minimised 
           ----
            Returns:
                the entry of results_list that is optimal
        '''

        ordered_results = []
        use_complexity = True

        for i, result in enumerate(results_list):
            results_folds = result['results folds']
            if 'complexity' not in result.keys():
                use_complexity = False
            cv_mean, cv_std_err = self._calculate_cv_mean_std_err(results_folds, objective_function)
            ordered_results.append((cv_mean, cv_std_err, i))

        # sort the items in ascending order by value of the cv_mean, the higher 
        # value of the cv_mean the better is the performance of the model
        ordered_results = list(sorted(ordered_results, key = lambda tup: tup[0]))
        if not maximum:
            ordered_results = list(reversed(ordered_results))

        idx_best_result = ordered_results[-1][-1]

        if regularisation == '1 SE':
            best_SE = ordered_results[-1][1]
            best_obj_val = ordered_results[-1][0]

            min_complexity = results_list[idx_best_result]['complexity'] if use_complexity else len(results_list[idx_best_result]['features'])
            current_obj_val = best_obj_val

            i = len(ordered_results)-1

            # look for the model with smallest complexity, s.t. its objective value
            # is still in the 1 SE range of the true optimal value and it has the 
            # highest score among all models with same complexity
            sign = 1 if maximum else -1
            while i >= 0 and sign*ordered_results[i][0] >= sign*best_obj_val - best_SE:
                # change to the corresponding index in the results_list
                i_results = ordered_results[i][-1]
                new_min_complexity = results_list[i_results]['complexity'] if use_complexity else len(results_list[i_results]['features'])

                if new_min_complexity < min_complexity:
                    idx_best_result = i_results
                    min_complexity = new_min_complexity

                i -= 1

        elif type(regularisation) in [int, float]:
            # choose the best performing model that does not exceed the given
            # regularisation threshold; if no such model exists, then the worst 
            # is chosen
            i = 0
            for j, res in enumerate(ordered_results):
                current_complexity = results_list[res[-1]]['complexity'] if use_complexity else len(results_list[res[-1]]['features'])
                if current_complexity <= regularisation:
                    i = j

            idx_best_result = ordered_results[i][-1]
            
        optimal_model = results_list[idx_best_result]

        return optimal_model


    def find_optimal_models(self, 
                            objective_function: t.Callable, 
                            regularisation: None | str | int | float = None,
                            maximum: bool = True
                            ) -> None:
        '''
            wrapper for `_find_optimal_model_in_list` that performs this for all
            models studied in the model selection procedure
           ----
            objective_function:
                the objective that is used to rank the models according to their
                performance
            regularisation:
                if not None, it can be either '1 SE' (one-standard-error-
                regularisation) or an integer/float (specifies the maximum level 
                of complexity that should not be exceeded)
            maximum:
                whether the objective function needs to be maximised or minimised 
        '''
        

        self.optimal_models = []

        for model_results in self.results.values():
            self.optimal_models.append(self._find_optimal_model_in_list(model_results, objective_function, regularisation, maximum))


    def summary(self):
        '''
            prints a summary output to the console
        '''

        if len(self.optimal_models) < 2:
            print('--- Global evaluation ---')
        
        else:
            print('--- Local evaluation ---')

        for opt_model in self.optimal_models:
            print(f'MODEL SELECTION (with {len(self.results)} models)\n')
            print('Best model: `{}`'.format(opt_model['model name']))

            max_len_feature_name = max([len(name) for name in opt_model['features']] + [len('opt. threshold')])

            title_coefficients = 'coefficients ' + '.' * (max_len_feature_name - len('coefficients') + 3) + ' CV est. +- CV SE'
            print(';– ' + title_coefficients)
            print('|––' + '–' * len(title_coefficients))

            for i, feature_name in enumerate(opt_model['features']):
                coeff_est, coeff_se = self._calculate_cv_mean_std_err(opt_model['results folds'], 
                                                                      objective_function=lambda coeffs: coeffs[0][i], 
                                                                      kind='coefficients'
                                                                     )
                print('|– ' + feature_name + ' ' + '.' * (max_len_feature_name - len(feature_name) + 3) + ' '*(coeff_est >= 0) + f' {coeff_est:.3f} +- {coeff_se:.3f}')
            
            if self.model_type == 'classification':
                theta_est, theta_se = self._calculate_cv_mean_std_err(opt_model['results folds'], 
                                                                    kind='threshold'
                                                                    )
                print(';– opt. threshold ' + '.' * (max_len_feature_name - len('opt. threshold') + 3) + ' '*(theta_est >= 0) + f' {theta_est:.3f} +- {theta_se:.3f}')

            metrics_names = []

            for m in opt_model['results folds']:
                metrics_names += list(m.metrics_predict.keys())
                
            metrics_names = set(metrics_names)
            max_len_name = max([len(metric_name) for metric_name in metrics_names])
            
            title_metrics = 'Avg. CV metric'
            title_values = 'Values: CV mean +- CV SE'
            title = title_metrics + max(0, max_len_name-len(title_metrics)) * ' '\
                + ' | ' + title_values

            max_len_name = max(max_len_name, len(title_metrics))

            print('\n' + title)
            print('–' * len(title))

            for metric_name in sorted(metrics_names):
                cv_mean, cv_se = self._calculate_cv_mean_std_err(opt_model['results folds'], objective_function = lambda metrics: metrics[metric_name])
                print(metric_name + max(0, max_len_name-len(metric_name)) * ' ' + f' | {cv_mean:.3f} +- {cv_se:.3f}')

            print()
    
    def produce_latex_table(self, model_name, file_name, dpc=None):
        for i, opt_model in enumerate(self.optimal_models):
            if model_name != opt_model['model name']:
                continue

            with open(file_name + str(i) + '.txt', 'w') as out_file:
                # header
                out_file.write(r'''    % table containing the estimate of the model coefficients
    \begin{subfigure}[t]{\textwidth}
    \begin{tabular}{p{4cm}S[table-format=3.3(3)]}
        \toprule
        Coefficients & {CV Estimate ($\pm$ SE)}\\
        \midrule'''
                )

                # first line in coefficient part
                n_coeffs = len(opt_model['features']) + 2

                for j, feature_name in enumerate(opt_model['features']):
                    coeff_est, coeff_se = self._calculate_cv_mean_std_err(opt_model['results folds'], 
                                                                        objective_function=lambda coeffs: coeffs[0][j], 
                                                                        kind='coefficients'
                                                                        )
                    out_file.write('\n        ' + feature_name + ' & ' + f'{coeff_est:.3f}' + r' \pm' + f' {coeff_se:.3f}' + r'\\')

                # out_file.write intercept
                intercept_est, intercept_se = self._calculate_cv_mean_std_err(opt_model['results folds'], 
                                                                    kind='intercept'
                                                                    )
                out_file.write('\n        Intercept & ' + f'{intercept_est:.3f}' + r' \pm' + f' {intercept_se:.3f}' + r'\\')
                
                # out_file.write threshold
                theta_est, theta_se = self._calculate_cv_mean_std_err(opt_model['results folds'], 
                                                                    kind='threshold'
                                                                    )
                out_file.write('\n' + r'        Optimal Threshold $\theta^*$ & ' + f'{theta_est:.3f}' + r' \pm' + f' {theta_se:.3f}' + r'\\' + '\n')


                out_file.write(r'''        \bottomrule
    \end{tabular}
    \end{subfigure}

    \vspace{1em}

    % table containing threshold independent metrics
    \begin{subfigure}[t]{\textwidth}
    \begin{tabular}{p{4cm}S[table-format=3.3(3)]}
        \toprule
        Metrics indep. of $\theta$ & {CV Estimate ($\pm$ SE)} \\
        \midrule''')

                # moving on to printing the performance metrics
                threshold_independent_metrics = [r'$AUC$',
                                                r'$\rho_{n; X, Y_{I_c}}$ (all)', 
                                                r'$\rho_{n; X, Y_{I_c}}$ (diseased)'
                                                ]

                for name in threshold_independent_metrics:
                    cv_mean, cv_se = self._calculate_cv_mean_std_err(opt_model['results folds'], objective_function = lambda metrics: metrics[name])
                    out_file.write('\n       ' + name + f' & {cv_mean:.3f}' + r' \pm' + f' {cv_se:.3f}' + r'\\')

                out_file.write('\n')
                out_file.write(r'''        \bottomrule
    \end{tabular}
    \end{subfigure}

    \vspace{1em}

    % table containing threshold dependent metrics
    \begin{subfigure}[t]{\textwidth}
    \begin{tabular}{lS[table-format=3.3(3)]S[table-format=3.3(3)]}
        \toprule
        \shortstack{Metrics\\depend. on $\theta$} & {\shortstack{$\theta = \theta^*$\\CV Est. ($\pm$ SE)}} & {\shortstack{$\theta = 0$\\CV Est. ($\pm$ SE)}}\\
        \midrule''')

                threshold_dependent_metrics = [(r'$ACC(\theta)$', r'$ACC(0)$'), 
                                (r'$ACC_b(\theta)$', r'$ACC_b(0)$'), 
                                (r'$Spec(\theta)$', r'$Spec(0)$'), 
                                (r'$Sens(\theta)$', r'$Sens(0)$')
                                ]

                for name_t, name_zero in threshold_dependent_metrics:
                    cv_mean_t, cv_se_t = self._calculate_cv_mean_std_err(opt_model['results folds'], objective_function = lambda metrics: metrics[name_t])
                    cv_mean_zero, cv_se_zero = self._calculate_cv_mean_std_err(opt_model['results folds'], objective_function = lambda metrics: metrics[name_zero])
                    out_file.write('\n        ' + name_t + f' & {cv_mean_t:.3f}' + r' \pm' + f' {cv_se_t:.3f}' + f' & {cv_mean_zero:.3f}' + r' \pm' + f' {cv_se_zero:.3f}' + r'\\')

                out_file.write('\n')
                out_file.write(r'''        \bottomrule
    \end{tabular}
    \end{subfigure}''')
    
    def plot_scores_best_model(self, 
                               deformation_pattern_classes, 
                               figsize=(5, 5),
                               fig_name='optimal model', 
                               dpi=100, 
                               violin_plot=True,
                               plot_hlines=False
                               ):

        for opt_model in self.optimal_models:
            scores_predict = [m.scores_predict for m in opt_model['results folds']]
            classes_predict = [m.classes_predict for m in opt_model['results folds']]
            indices_predict = [m.indices_predict for m in opt_model['results folds']]
            optimal_thresholds = np.array([m.optimal_threshold for m in opt_model['results folds']])

            folds = self.folds
            if len(folds) < 2:
                folds = [self.keys_to_patients]

            scores = {}
            scores[fig_name] = {'control': {}, 'disease': {}}

            for sp, cp, ip in zip(scores_predict, classes_predict, indices_predict):
                for s, c, i in zip(sp, cp, ip):
                    scores[fig_name][c][i] = s

            # handle the case, where one of the thresholds happens to be \infty 
            # and would lead to the mean being shifted to somewhere nonsensical
            mask_infs = np.isinf(optimal_thresholds)
            optimal_thresholds = optimal_thresholds[~mask_infs]

            hlines = None
            if plot_hlines == 'mean':
                hlines = {fig_name: np.mean(optimal_thresholds)}
            elif plot_hlines == 'each':
                hlines = {fig_name: list(optimal_thresholds)}
            
            plot_scores(scores, 
                        figsize=figsize, 
                        dpi=dpi, 
                        n_cols=1, 
                        deformation_pattern_classes=deformation_pattern_classes, 
                        violin_plot=violin_plot,
                        hlines=hlines
                        )

    def plot_roc_curves_best_model(self,
                               model_name,
                               figsize=(5, 5),
                               fig_name='optimal model', 
                               dpi=100,
                               savefig=None,
                               show_intercept=True
                               ):

        for k, opt_model in enumerate(self.optimal_models):
            if model_name != opt_model['model name']:
                continue

            roc_predict = [np.array(m.roc_curve_values_predict) for m in opt_model['results folds']]
            optimal_thresholds_roc_space = np.array([[1 - m.metrics_predict[r'$Spec(\theta)$'],
                                                      m.metrics_predict[r'$Sens(\theta)$']] \
                                                     for m in opt_model['results folds']
                                                    ]
                                                   )
            zero_thresholds_roc_space = np.array([[1 - m.metrics_predict[r'$Spec(0)$'],
                                                   m.metrics_predict[r'$Sens(0)$']] \
                                                  for m in opt_model['results folds']
                                                 ]
                                                )
            thresholds_roc_space = [m.thresholds_predict for m in opt_model['results folds']]

            # threshold averaging for ROC curve
            all_sorted_thresholds = list(sorted(sum([list(t_values) for t_values in thresholds_roc_space], [])))
            avg_roc_curve = np.zeros((len(all_sorted_thresholds), 2))
            se_roc_curve = np.zeros((len(all_sorted_thresholds), 2))
            n_roc_curves = len(roc_predict)

            for i, t in enumerate(all_sorted_thresholds):
                temp_values = []
                for j in range(n_roc_curves):
                    # choose last largest threshold theta s.t. theta <= t
                    possible_indices = np.arange(len(thresholds_roc_space[j]))[thresholds_roc_space[j] <= t]
                    idx_threshold_choice = -1 if len(possible_indices) == 0 else possible_indices[0]
                    temp_values.append(roc_predict[j][idx_threshold_choice])

                avg_roc_curve[i] = sum(temp_values) / n_roc_curves
                se_roc_curve[i] = np.sqrt(sum([(v - avg_roc_curve[i])**2 for v in temp_values]) / ((n_roc_curves-1)*n_roc_curves))

            plt.figure(figsize=figsize, dpi=dpi)
            plt.title(fig_name)
            show_legend = True

            plt.plot([0, 1], [0, 1], ls='--', color='chocolate', alpha=0.7)
            plt.plot(avg_roc_curve[:, 0], avg_roc_curve[:, 1], color='black', label='avg. ROC curve')

            for roc_curve, opt_thresh_coords, z_thresh_coords in zip(roc_predict, 
                                                                     optimal_thresholds_roc_space, 
                                                                     zero_thresholds_roc_space
                                                                     ):
                if show_legend:
                    plt.plot(roc_curve[:, 0], roc_curve[:, 1], color='darkcyan', alpha=0.3, label='ROC curves')
                    plt.scatter(opt_thresh_coords[0], opt_thresh_coords[1], marker='*', color='darkcyan', label=r'$\theta = \theta^*$')
                    if show_intercept:
                        plt.scatter(z_thresh_coords[0], z_thresh_coords[1], marker='o', color='lightseagreen', label=r'$\theta = 0$')
                    show_legend = False
                else:
                    plt.plot(roc_curve[:, 0], roc_curve[:, 1], color='darkcyan', alpha=0.3)
                    plt.scatter(opt_thresh_coords[0], opt_thresh_coords[1], marker='*', color='darkcyan')
                    if show_intercept:
                        plt.scatter(z_thresh_coords[0], z_thresh_coords[1], marker='o', color='lightseagreen')

            plt.xlabel(r'$1 - Spec$')
            plt.ylabel(r'$Sens$')
            plt.legend()   
            plt.tight_layout() 
            if savefig is not None:
                plt.savefig(savefig + str(k) + '.png')        
            plt.show()


    def performance_complexity_plot(self, model_name, objective_function, maximum, dpi=100, figsize=(3.5, 3.5)):
        cv_means = []
        cv_std_errs = []
        complexity = []

        for result in self.results[model_name]:
            results_folds = result['results folds']
            if 'complexity' not in result.keys():
                complexity.append(len(result['features']))
            else:
                complexity.append(result['complexity'])
            cv_mean, cv_std_err = self._calculate_cv_mean_std_err(results_folds, objective_function)
            cv_means.append(cv_mean)
            cv_std_errs.append(cv_std_err)

        one_se_model = self._find_optimal_model_in_list(self.results[model_name], objective_function, regularisation='1 SE', maximum=maximum)
        one_se_cv_mean, _ = self._calculate_cv_mean_std_err(one_se_model['results folds'], objective_function)
        one_se_complexity = one_se_model['complexity'] if 'complexity' in one_se_model.keys() else len(one_se_model['features'])
        
        cv_means = np.array(cv_means)
        cv_std_errs = np.array(cv_std_errs)
        i_extreme = np.argmax(cv_means) if maximum else np.argmin(cv_means)

        plt.figure(figsize=figsize, dpi=dpi)

        plt.fill_between(complexity, 
                        cv_means-cv_std_errs, 
                        cv_means+cv_std_errs, 
                        alpha=0.4,
                        label='CV standard error')
        plt.plot(complexity, 
                cv_means,
                color='tab:blue',
                label='CV estimate')
        plt.scatter(one_se_complexity, 
                    one_se_cv_mean, 
                    marker='.', 
                    color='tab:blue', 
                    s=100, 
                    label='one-SE-regularised-model')
        plt.scatter(complexity[i_extreme], 
                    cv_means[i_extreme], 
                    marker='*', 
                    color='tab:blue', 
                    s=100, 
                    label='best model')
        plt.legend()
        ax = plt.gca()
        ax.set_xticks(np.arange(complexity[-1], step=1), minor=True)
        ax.set_xticks(np.arange(complexity[-1], step=10))
        plt.grid(visible=True, which='major', linestyle='-')
        plt.grid(visible=True, which='minor', linestyle='--')

        return ax


def latex_table_high_level_summary(model_selection_instances):
    print(r'''\begin{tabular}{lcccc}
    \toprule
    Model (Dataset) & $ACC_b(\theta^*)$ & $Spec(\theta^*)$ & $Sens(\theta^*)$ & $N_\text{features}$\\
    \midrule''')
    
    for instance_name, instance in model_selection_instances:
        for opt_model in instance.optimal_models: 
            
            metrics_names = [r'$ACC_b(\theta)$', r'$Spec(\theta)$', r'$Sens(\theta)$']
            
            print('    ' + opt_model['model name'] + f' ({instance_name})', end='')
            
            for metric_name in metrics_names:
                cv_mean, cv_se = instance._calculate_cv_mean_std_err(opt_model['results folds'], objective_function = lambda metrics: metrics[metric_name])
                print(f' & {cv_mean:.3f} ' + r'$\pm$' + f' {cv_se:.3f}', end='')

            n_feat = len(opt_model['features'])
            print(f' & {n_feat}' + r'\\')

    print(r'''    \bottomrule
\end{tabular}''')