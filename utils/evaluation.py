import itertools
import typing as t
import inspect

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import dtw

from .data import EchocardiographyDataCollection
from .alignment import align_dtw, align_fdasrsf
from .geometry import area_between_warping_curves
from .numerical_methods import differentiate, integrate
from .mast_classification import MAST_TYPE_COLORS


class SingleEvalMethods():

    def _evaluate(C: EchocardiographyDataCollection,
                 evaluation_function: t.Callable,
                 cycles: int | list = 0,
                 align: None | t.Callable = None
                 ) -> dict:
        '''
            A general framework for evaluating methods on the echocardiography data.
           ----
            C:
                an EchochardiographyDataCollection class instance
            evaluation_function:
                a method that computes a score for each patient based on their longitudinal strain data;
                the function signature changes depending on what alignment method is used;
                call signature has to be ('time', 'curves') or ('ECD')
            cycles:
                specifies which of the recorded cycles should be evaluated
            align:
                if not None, this is the alignment procedure to be used to align (combinations of) the curves
           ----
            Returns:
                a dictionary of scores for each group and each patient
        '''

        scores = {}

        if type(cycles) == int:
            cycles = [cycles]
        
        for cycle in cycles:
            for group in C.collection.keys():
                if group not in scores.keys():
                    scores[group] = {}
                for patient in C.collection[group].keys():
                    ECD = C.collection[group][patient]

                    if cycle >= len(ECD.time_normalised_data['Longitudinal Strain-Endo']):
                        continue

                    data = ECD.time_normalised_data['Longitudinal Strain-Endo'][cycle].to_numpy()
                    time = data[:, -1]
                    curves = data[:, :-1]

                    # skip the current individual if the recorded cycle is too small
                    if time[-1] - time[0] < 0.8:
                        continue

                    eval_function_signature = inspect.signature(evaluation_function).parameters.keys()

                    # calculate the score
                    score = None
                    if align is not None:
                        alignment_output = align(time, curves)

                        # now filter the output for the arguments needed in the evaluation function ...
                        filtered_output = {arg: alignment_output[arg] for arg in eval_function_signature}

                        # ... check if the method and the alignment are compatible ...
                        if not (set(filtered_output.keys()) == set(eval_function_signature)):
                            raise NameError('The arguments of the evaluation_method and the output of the alignment procedure are not compatible.')

                        # ... and evaluate
                        score = evaluation_function(**filtered_output)

                    elif set(['time', 'curves']) == set(eval_function_signature):
                        score = evaluation_function(time, curves)

                    elif set(['ECD']) == set(eval_function_signature):
                        score = evaluation_function(ECD)

                    else:
                        raise RuntimeError('`evaluation_function` does not have a compatible signature.')
                    

                    if len(cycles) > 1:
                        if patient not in scores[group].keys():
                            scores[group][patient] = []
                        scores[group][patient].append(score)

                    else:
                        scores[group][patient] = score

        return scores


    ### existing methods

    def gls(self, time, curves):
        return np.min(np.mean(curves, axis=1))

    def systolic_gls(self, ECD):
        # GLS on first cycle
        data = ECD.data['Longitudinal Strain-Endo'].to_numpy()
        time = data[:, -1]
        n = np.sum(time <= ECD.end_systole_ms - ECD.end_diastole_ms)
        end_systole_idx = np.arange(n)[-1]
        return np.mean(data[end_systole_idx, :-1])

    def mechanical_dispersion(self, time, curves):
        min_indices = np.argmin(curves, axis=0)
        min_times = time[min_indices]
        return np.std(min_times, ddof=1)


    ### absolute error methods

    def integrated_mean_absolute_error(self, time, curves, mean_curve=None):
        if mean_curve is None:
            mean_curve = np.mean(curves, axis=1)
        mean_abs_err = np.mean(np.abs(curves - mean_curve[:, None]), axis=1)
        return integrate(time, mean_abs_err)
    
    def integrated_max_absolute_error(self, time, curves, mean_curve=None):
        if mean_curve is None:
            mean_curve = np.mean(curves, axis=1)
        mean_abs_err = np.max(np.abs(curves - mean_curve[:, None]), axis=1)
        return integrate(time, mean_abs_err)

    def integrated_mean_permutation(self, time, curves):
        comb = []
        for i, j in itertools.combinations(range(curves.shape[1]), 2):
            comb.append(np.abs(curves[:, i] - curves[:, j]))
        max_curve = np.mean(np.stack(comb), axis=0)
        return integrate(time, max_curve)
    
    def integrated_max_permutation(self, time, curves):
        comb = []
        for i, j in itertools.combinations(range(curves.shape[1]), 2):
            comb.append(np.abs(curves[:, i] - curves[:, j]))
        max_curve = np.max(np.stack(comb), axis=0)
        return integrate(time, max_curve)


    ### time difference methods

    def _transition_time(self, d_time, d_curves, criterion):
        v0_t0 = (0.0, 0.0)
        t_diffs = []
        max_diffs = []
        
        value_time_pairs = []
        old_d_curves_smaller_0 = False
        crossed_0 = False

        for i, d_curves_smaller_0 in enumerate(d_curves < 0):
            if not d_curves_smaller_0 and old_d_curves_smaller_0 and not crossed_0:
                v0_t0 = (d_curves[i], d_time[i])
                value_time_pairs.append((d_curves[i], d_time[i]))
                crossed_0 = True

            elif crossed_0: 
                if not d_curves_smaller_0:
                    value_time_pairs.append((d_curves[i], d_time[i]))

                if d_curves_smaller_0 or i >= len(d_time)-1:
                    if criterion == 'min_to_inflection':
                        max_val, max_t = max(value_time_pairs, key=lambda t: t[0])
                        max_diffs.append(float(max_val - v0_t0[0]))
                        t_diffs.append(float(max_t - v0_t0[1]))
                    elif criterion == 'min_to_max':
                        max_diffs.append(float(d_curves[i] - v0_t0[0]))
                        t_diffs.append(float(d_time[i] - v0_t0[1]))
                    
                    value_time_pairs = []
                    crossed_0 = False
                
            old_d_curves_smaller_0 = d_curves_smaller_0

        if len(max_diffs) > 0:
            # TODO: Check if this works better than using the key `lambda t: t[1]`
            return max([tup for tup in zip(max_diffs, t_diffs)], key=lambda t: t[0])[1]
        
        return d_time[-1] - d_time[0]

    def MTI(self, time, curves):
        d_time, d_curves = differentiate(time, curves)
        ti = [self._transition_time(d_time, d_curves[:, i], criterion='min_to_inflection') for i in range(d_curves.shape[1])]
        return np.mean(ti)

    def MTM(self, time, curves):
        d_time, d_curves = differentiate(time, curves)
        tm = [self._transition_time(d_time, d_curves[:, i], criterion='min_to_max') for i in range(d_curves.shape[1])]
        return np.mean(tm)
    
    def MaxTI(self, time, curves):
        d_time, d_curves = differentiate(time, curves)
        ti = [self._transition_time(d_time, d_curves[:, i], criterion='min_to_inflection') for i in range(d_curves.shape[1])]
        return np.max(ti)

    def MaxTM(self, time, curves):
        d_time, d_curves = differentiate(time, curves)
        tm = [self._transition_time(d_time, d_curves[:, i], criterion='min_to_max') for i in range(d_curves.shape[1])]
        return np.min(tm)


    def _TI_segmental(self, time, curves, segment=0):
        d_time, d_curves = differentiate(time, curves)
        return self._transition_time(d_time, d_curves[:, segment], criterion='min_to_inflection')
    
    def TI_segmental(self, segment):
        return lambda time, curves: self._TI_segmental(time, curves, segment=segment)

    def _TM_segmental(self, time, curves, segment=0):
        d_time, d_curves = differentiate(time, curves)
        return self._transition_time(d_time, d_curves[:, segment], criterion='min_to_max')
    
    def TM_segmental(self, segment):
        return lambda time, curves: self._TM_segmental(time, curves, segment=segment)

    ### total variation methods

    def _total_variation_measure(self, time, curves, kind='mean', segment=0):
        d_curves = (curves[1:] - curves[:-1])/(time[1:, None] - time[:-1, None])
        d_time = time[:-1]

        tvs = [integrate(d_time, np.abs(d_curves[:, i])) for i in range(d_curves.shape[1])]

        if kind == 'max':
            return np.max(tvs)
        
        if kind == 'mean':
            return np.mean(tvs)

        if kind == 'rms':
            return np.mean(np.array(tvs)**2)

        if kind == 'minmax':
            return np.max(tvs) - np.min(tvs)
        
        if kind == 'segmental':
            return tvs[segment]
        
        else:
            raise ValueError(f'Kind {kind} is not a valid method.')

    def MTV(self, time, curves):
        return self._total_variation_measure(time, curves, kind='mean')

    def RMSTV(self, time, curves):
        return self._total_variation_measure(time, curves, kind='rms')

    def MaxTV(self, time, curves):
        return self._total_variation_measure(time, curves, kind='max')
    
    def MinMaxTV(self, time, curves):
        return self._total_variation_measure(time, curves, kind='minmax')

    def TV_segmental(self, segment):
        return lambda time, curves: self._total_variation_measure(time, curves, kind='segmental', segment=segment)

    def MTV_signed(self, time, curves):
        d_time, d_curves = differentiate(time, curves, method='default')
        mask = np.all(d_curves > 0, axis=1) | np.all(d_curves < 0, axis=1) | np.all(d_curves == 0, axis=1)
        return integrate(d_time, mask * np.mean(np.abs(d_curves), axis=1))


    ### other methods

    def basal_variance(self, time, curves):
        var = 1/(curves.shape[1]-2) * np.sum((curves[:, 1:] - curves[:, 0][:, None])**2, axis=1)
        return integrate(time, var)


    def mean_positive_value_penalty(self, time, curves):
        positive_part = curves * (curves > 0)
        return np.mean([integrate(time, positive_part[:, i]) for i in range(curves.shape[1])])
    
    def max_positive_value_penalty(self, time, curves):
        positive_part = curves * (curves > 0)
        return np.max([integrate(time, positive_part[:, i]) for i in range(curves.shape[1])])

    def _positive_value_penalty_segmental(self, time, curves, segment=0):
        positive_part = curves * (curves > 0)
        return integrate(time, positive_part[:, segment])

    def positive_value_penalty_segmental(self, segment):
        return lambda time, curves: self._positive_value_penalty_segmental(time, curves, segment)


    def mean_positive_peak_penalty(self, time, curves):
        return np.mean(np.max(curves, axis=0))

    def max_positive_peak_penalty(self, time, curves):
        return np.max(curves)

    def _positive_peak_penalty_segmental(self, time, curves, segment=0):
        return np.max(curves[:, segment])

    def positive_peak_penalty_segmental(self, segment):
        return lambda time, curves: self._positive_peak_penalty_segmental(time, curves, segment) 


    def peak_strain(self, time, curves):
        return np.min(curves)

    def mean_peak_strain(self, time, curves):
        return np.mean(np.min(curves, axis=0))

    def _peak_strain_segmental(self, time, curves, segment=0):
        return np.min(curves[:, segment])
    
    def peak_strain_segmental(self, segment):
        return lambda time, curves: self._peak_strain_segmental(time, curves, segment)


    def area_below_peak_strain(self, time, curves):
        min_indices = np.argmin(curves, axis=0)
        scaled_curves = curves / np.max(np.abs(curves), axis=0)
        return np.mean([integrate(time[:min_indices[i]], scaled_curves[:min_indices[i], i]) for i in range(scaled_curves.shape[1])])

    def _area_below_peak_strain_segmental(self, time, curves, segment=0):
        curve = curves[:, segment]
        min_idx = np.argmin(curve)
        scaled_curve = curve / np.max(np.abs(curve))
        return integrate(time[:min_idx], scaled_curve[:min_idx])

    def area_below_peak_strain_segmental(self, segment):
        return lambda time, curves: self._area_below_peak_strain_segmental(time, curves, segment)



class MultiEvalMethods():

    def _evaluate(C: EchocardiographyDataCollection,
                 evaluation_function: t.Callable
                ) -> dict:
        '''
            A general framework for evaluating methods on the echocardiography data.
           ----
            C:
                an EchochardiographyDataCollection class instance
            evaluation_function:
                a method that computes a score for each patient based on their longitudinal strain data;
                the function signature changes depending on what alignment method is used;
                call signature has to be ('C')
           ----
            Returns:
                a dictionary of scores for each group and each patient
        '''

        raise NotImplementedError

    def pca(self):
        raise NotImplementedError


class EvaluationMethods(SingleEvalMethods, MultiEvalMethods):

    def __init__(self):
        '''
            a class combining the functionality of SingleEvalMethods and MultiEvalMethods
        '''
        super().__init__()
        self.scores = None

    def evaluate(self,
                 C,
                 evaluation_function,
                 cycles=0,
                 align=None
                 ):

        # check if function is a member of the SingleEvalMethods class
        if set(inspect.signature(evaluation_function).parameters.keys()) == set(['time', 'curves'])\
            or set(inspect.signature(evaluation_function).parameters.keys()) == set(['ECD'])\
            or align is not None:
            self.scores = SingleEvalMethods._evaluate(C=C, 
                                                     evaluation_function=evaluation_function,
                                                     cycles=cycles,
                                                     align=align
                                                    )
                                            
        # check if function is a member of the MultiEvalMethods class
        elif set(inspect.signature(evaluation_function).parameters.keys()) == set(['C']):
            self.scores = MultiEvalMethods._evaluate(C=C, 
                                                    evaluation_function=evaluation_function, 
                                                    cycles=cycles,
                                                    align=align
                                                   )            

        else:
            raise RuntimeError('`evaluation_function` does not have a valid function signature.')


class AlignmentEvaluationMethods(EvaluationMethods):

    def __init__(self, alignment_function: t.Callable):
        '''
            a class that allows for the evaluation of aligned data
           ----
            alignment_function:
                a function that aligns the the data; see align argument  in 
                EvaluationMethods.evaluation and the child-classes below for call 
                signature and output format
        '''
        super().__init__()
        self.alignment_function = alignment_function
    
    def align(self, time, curves):
        return self.alignment_function(time, curves)

    def evaluate(self, C, evaluation_function):
        super().evaluate(C=C, 
                         evaluation_function=evaluation_function, 
                         align=self.align
                         )


class DtwMethods(AlignmentEvaluationMethods):

    def __init__(self, 
                 step_pattern: dtw.StepPattern, 
                 alignment_type: str, 
                 means_time_tup: None | tuple[np.ndarray, np.ndarray] = None, 
                 dba_args: dict = None
                ) -> None:
        '''
            a class that allows for the evaluation of dtw-aligned data
           ----
            step_pattern:
                the step pattern used by the dtw.dtw function
            alignment_type: 
                specifies the type of alignment that should be used; possible options are
                "combinations" (default), "permutations", "mean", "derivative", "distance measure",
                "integration", "segmental mean"
            means_time_tup:
                segmental means and corresponding time grids
            dba_args:
                arguments for calling the function `dtw_barycentric_averaging` when
                alignment_type == "mean" 
        '''
        alignment_function = lambda time, curves: align_dtw(time=time, 
                                                            curves=curves, 
                                                            step_pattern=step_pattern, 
                                                            alignment_type=alignment_type,
                                                            means_time_tup=means_time_tup, 
                                                            dba_args=dba_args
                                                           )
        super().__init__(alignment_function=alignment_function)


    ### DTW alignment specific evaluation methods

    def IV_dtw(self, time, time_warping_1, time_warping_2):
        areas = []
        for t in time_warping_1 + time_warping_2:
            areas.append(area_between_warping_curves(time, t, np.array([[d, d] for d in time])))
        
        return np.mean(areas)

    def IV_dtw_combinations(self, time, time_warping_1, time_warping_2):
        areas = []
        for t, t_hat in zip(time_warping_1, time_warping_2):
            areas.append(area_between_warping_curves(time, t, t_hat))
        
        return np.mean(areas)

    def dtw_mean_distance(self, distances):
        return np.mean(distances)


class FdasrsfMethods(AlignmentEvaluationMethods):

    def __init__(self):
        '''
            a class that allows for the evaluation of fdasrsf-aligned data
        '''
        super().__init__(alignment_function=align_fdasrsf)


    ### FDASRSF alignment specific evaluation methods

    def use_aligned_curves(self, method):
        return lambda time, aligned_curves: method(time, aligned_curves)

    def IV(self, time, time_warping):
        v = np.var(time[:, None] - time_warping, ddof=1, axis=1)
        return integrate(time, v)


'''

def evaluate(C: EchocardiographyDataCollection,
             evaluation_function: t.Callable,
             cycle: int = 0,
             align: None | t.Callable = None,
             verbose: bool = False
            ) -> dict:
    scores = {}
    
    for group in C.collection.keys():
        if verbose:
            print(f'group: {group}')
        scores[group] = {}
        for patient in C.collection[group].keys():
            ECD = C.collection[group][patient]
            data = ECD.time_normalised_data['Longitudinal Strain-Endo'][cycle].to_numpy()
            time = data[:, -1]
            curves = data[:, :-1]

            # calculate the score
            score = None
            if align is not None:
                alignment_output = align(time, curves)

                # now filter the output for the arguments needed in the evaluation function ...
                function_signature = inspect.signature(evaluation_function).parameters.keys()
                filtered_output = {arg: alignment_output[arg] for arg in function_signature}

                # ... check if the method and the alignment are compatible ...
                if not (set(filtered_output.keys()) == set(function_signature)):
                    raise NameError('The arguments of the evaluation_method and the output of the alignment procedure are not compatible.')

                # ... and evaluate
                score = evaluation_function(**filtered_output)
            else:
                score = evaluation_function(time, curves)

            if verbose:
                print(f'    ID {patient}: {score :.3f}')
            scores[group][patient] = score
    
        # summary
        if verbose:
            scores_group = [v for v in scores[group].values()]
            print(f'    -----\n    avg. score: {np.mean(scores_group):.3f}', end='')
            if len(scores_group) > 1:
                print(f' +- {np.std(scores_group, ddof=1):.3f}\n')
            else:
                print('\n')

    return scores

'''



def plot_scores(scores, 
                figsize=None, 
                dpi=100, 
                n_cols=4, 
                savefig=None, 
                selection=None, 
                deformation_pattern_classes=None,
                violin_plot=False,
                hlines=None
               ):

    colors = {'control': 'tab:blue',
              'disease': 'tab:red'
             }

    shifts_def_patterns = {'Type-I': -0.1,
                           'Type-II': 0.0,
                           'Type-III': 0.1}

    shifts_def_patterns_violin = {'Type-I': -0.2,
                                    'Type-II': 0.0,
                                    'Type-III': 0.2}
    
    groups = []
    x_groups = {}
    x_ticks = {}
    y_groups = {}
    
    for i, method in enumerate(scores):
        for group in scores[method]:
            patient_values = scores[method][group].values()
            
            if group not in groups:
                x_groups[group] = [i] * len(patient_values)
                x_ticks[group] = [method] * len(patient_values)
                y_groups[group] = [v for v in patient_values]
                groups.append(group)
    
            else:
                x_groups[group] += [i] * len(patient_values)
                x_ticks[group] += [method] * len(patient_values)
                y_groups[group] += [v for v in patient_values]
    
    
    n_methods = len(scores.keys())
    n_cols = min(n_methods, n_cols)
    n_rows = (n_methods-1)//n_cols + 1
    
    fig, ax = plt.subplots(nrows=n_rows, 
                           ncols=n_cols,
                           figsize=(n_methods*3.0, n_methods*0.8) if figsize is None else figsize,
                           dpi=dpi)
    
    #fig.suptitle('Distribution of IV-score for different methods', fontweight='bold')
    
    for i, method in enumerate(scores):
        axs = None
        if n_methods <= 1:
            axs = ax
        elif n_rows <= 1:
            axs = ax[i]
        else:
            row = i//n_cols
            col = i%n_cols
            axs = ax[row, col]

        x_values = []
        x_ticks = []
        for j, group in enumerate(scores[method]):
            patient_values = scores[method][group].values()
            patient_keys = scores[method][group].keys()
    
            x_values += [j]
            x_ticks += [group]
    
            if len(patient_values) > 1:
                # give the labels for the first k subjects
                if type(selection) == int and selection > 0 and selection <= len(patient_keys)/2:
                    y = np.array([v for v in patient_values])

                    indices_outliers_low = np.argsort(y)[:selection]
                    indices_outliers_high = np.argsort(y)[-selection:]

                    keys_outliers = [list(scores[method][group].keys())[k] for k in indices_outliers_low]
                    keys_outliers += [list(scores[method][group].keys())[k] for k in indices_outliers_high]
                    values_outliers = [y[k] for k in indices_outliers_low] + [y[k] for k in indices_outliers_high]

                    texts = []
                    for k, (key, v) in enumerate(zip(keys_outliers, values_outliers)):
                        texts.append(axs.text(j, v, key, ha='right' if k%2 else 'left', va='center'))

                # plot the labels for a user specified selection of individuals
                elif type(selection) == dict:
                    keys = list(set(patient_keys) & set(selection))
                    y = np.array([v for v in patient_values])

                    keys_selection = []
                    values_selection = []

                    for k, v in zip(patient_keys, patient_values):
                        if k in selection:
                            keys_selection.append(k)
                            values_selection.append(v)

                    keys_selection = np.array(keys_selection)[np.argsort(values_selection)]
                    values_selection = np.sort(values_selection)

                    texts = []
                    for k, (key, v) in enumerate(zip(keys_selection, values_selection)):
                        texts.append(axs.text(j, v, key, ha='right' if k%2 else 'left', va='center'))
                               
            if deformation_pattern_classes is not None:
                dfc_arr = np.array([deformation_pattern_classes[group][patient] for patient in patient_keys])
                dfc_class_str = set(dfc_arr)

                if violin_plot:
                    for s in dfc_class_str:
                        mask_dfc_class = dfc_arr == s

                        if np.sum(mask_dfc_class) < 1:
                            continue

                        patient_values_dfc_class = np.array(list(patient_values))[mask_dfc_class]
                        shift = shifts_def_patterns_violin[s]

                        parts = axs.violinplot(patient_values_dfc_class, 
                                                positions = [j + shift],
                                                showmeans=True,
                                                widths=0.15,
                                                quantiles=[0.25, 0.75]
                                                )

                        for partname in ('cbars','cmins','cmaxes','cmeans', 'cquantiles'):
                            vp = parts[partname]
                            vp.set_edgecolor(MAST_TYPE_COLORS[s])

                        for pc in parts['bodies']:
                            pc.set_color(MAST_TYPE_COLORS[s])
                            pc.set_alpha(0.5)

                else:
                    for patient, v, def_pattern_class in zip(patient_keys, patient_values, dfc_arr):
                        axs.scatter(j + shifts_def_patterns[def_pattern_class], 
                                    v, 
                                    color=MAST_TYPE_COLORS[def_pattern_class], 
                                    alpha=0.7
                                )
            else:
                if violin_plot:
                    parts = axs.violinplot(patient_values, 
                                           positions = [j],
                                           showmeans=True,
                                           widths=0.4,
                                           quantiles=[0.25, 0.75]
                                           )

                    for partname in ('cbars','cmins','cmaxes','cmeans', 'cquantiles'):
                            vp = parts[partname]
                            vp.set_edgecolor(colors[group])

                    for pc in parts['bodies']:
                        pc.set_color(colors[group])
                        pc.set_alpha(0.5)
                else:
                    axs.scatter([j] * len(patient_values), 
                                    [v for v in patient_values],
                                    color=colors[group], 
                                    alpha=0.7
                            )
            
        axs.set_title(method)
        axs.set_xticks(x_values, x_ticks)
        axs.set_xlim([x_values[0]-1, x_values[-1]+1])

        if hlines is not None:
            if type(hlines[method]) == list:
                for hline in hlines[method]:
                    axs.axhline(y=hline, color='black', ls='--', alpha=0.7)    
            else:
                axs.axhline(y=hlines[method], color='black', ls='--')
    
    fig.tight_layout()
    
    if savefig is not None:
        fig.savefig(savefig)

    plt.show()


def plot_ordered_scores(C, scores, figsize, dpi=100, n_cols=10, title_type='ID', cycle=0, savefig=None):
    n_plots = sum([len(scores[group].keys()) for group in scores])

    groups = {
    'control': 'blue', 
    'disease': 'red'
    }

    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    n_rows = (n_plots-1)//n_cols + 1

    fig, axs = plt.subplots(figsize=figsize,
                            dpi=dpi,
                            nrows=n_rows, 
                            ncols=n_cols)

    scores_unordered = []

    for group in scores:
        for patient in scores[group]:
            scores_unordered.append((scores[group][patient], group, patient))
    
    scores_ordered = list(sorted(scores_unordered, key=lambda tup: tup[0]))

    for i, tup in enumerate(scores_ordered):
        score, group, patient = tup
        ECD = C.collection[group][patient]
        data = ECD.time_normalised_data['Longitudinal Strain-Endo'][cycle].to_numpy()
        seg_names = ECD.time_normalised_data['Longitudinal Strain-Endo'][cycle].columns
        time = data[:, -1]
        curves = data[:, :-1]
        
        ax = None
        if n_plots <= 1:
            ax = axs
        elif n_rows <= 1:
            ax = axs[i]
        else:
            row = i//n_cols
            col = i%n_cols
            ax = axs[row, col]

        if title_type == 'ID':
            ax.set_title(group + ' (ID '+ patient + ')')
        elif title_type == 'score':
            ax.set_title(group + f' (score = {score:.3f})')
        elif title_type == 'both':
            ax.set_title(group + ' (ID '+ patient + ',' + f'\nscore = {score:.3f})')
        
        for col in range(curves.shape[1]):
            ax.plot(time, 
                    curves[:, col], 
                    label=seg_names[col],
                    color=colours[col],
                    )
            
        #ax.set_xlabel(r'Relative time $t/T_i$')
        #ax.set_ylabel('Longitudinal Strain')
        ax.grid()
    
    fig.tight_layout()

    if savefig is not None:
        fig.savefig(savefig)
    plt.show() 

def plot_curves_gradient(C, scores, figsize, dpi=100, cycle=0):
    n_plots = sum([len(scores[group].keys()) for group in scores])

    scores_unordered = []

    for group in scores:
        for patient in scores[group]:
            scores_unordered.append((scores[group][patient], group, patient))
    
    scores_ordered = list(sorted(scores_unordered, key=lambda tup: tup[0]))
    scores_only = [tup[0] for tup in scores_ordered]

    norm = mcolors.Normalize(vmin=min(scores_only), vmax=max(scores_only))
    colours = cm.viridis(norm(scores_only))

    fig, axs = plt.subplots(2, 3, figsize=figsize)
    for i in range(len(scores_ordered)):
        group, patient = scores_ordered[i][1], scores_ordered[i][2]
        ECD = C.collection[group][patient]
        curves = ECD.time_normalised_data['Longitudinal Strain-Endo'][cycle].iloc[:, :-1]
        time = ECD.time_normalised_data['Longitudinal Strain-Endo'][cycle].iloc[:, -1]

        for s, seg in enumerate(curves.columns):
            if group == 'disease':
                axs[s//3, s%3].plot(time, 
                                    curves[seg],
                                    color=colours[i]
                                    )
            if group == 'control':
                axs[s//3, s%3].plot(time, 
                                    curves[seg],  
                                    ls='--',
                                    color=colours[i]
                                    )

            if i < 1:
                axs[s//3, s%3].set_title(seg)
                

    #plt.colorbar(scores_only)
    plt.show()

def plot_variations_across_cycles(scores, 
                                figsize=None, 
                                dpi=100, 
                                n_cols=4, 
                                savefig=None, 
                                selection=None, 
                                deformation_pattern_classes=None,
                                violin_plot=False):

    sd_scores = {}

    for i, method in enumerate(scores):
        sd_scores[method + ' SD'] = {}
        for group in scores[method]:
            sd_scores[method + ' SD'][group] = {}
            for patient, v in scores[method][group].items():
                if type(v) == list and len(v) > 1:
                    sd_scores[method + ' SD'][group][patient] = np.std(v, ddof=1)

    plot_scores(sd_scores, 
                figsize=figsize, 
                dpi=dpi, 
                n_cols=n_cols, 
                savefig=savefig, 
                selection=selection, 
                deformation_pattern_classes=deformation_pattern_classes,
                violin_plot=violin_plot)