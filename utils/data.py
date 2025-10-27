import os
import json
import copy
import typing as t
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline

from .constants import PATH_TO_CACHE, PATH_TO_CONTROL, PATH_TO_DISEASE,\
    PREC, PATH_TO_DEMOGRAPHIC_CONTROL, PATH_TO_DEMOGRAPHIC_DISEASE
from .tools import convert_echocardiography_xlsx_to_json, convert_demography_data_to_json

T = t.TypeVar('T')


class EchocardiographyData():
    data: dict
    location: str
    time_normalised_data: list
    bpm: float
    cycle_length_ms: float
    end_diastole_ms: float
    end_systole_ms: float
    demographics: dict
    
    means_stds: dict
    time_norm_means_stds: dict

    def __init__(self, 
                 json_data: dict, 
                 demographics: None | dict = None
                 ):
        '''
            a class for storing echocardiography data for a given individual
           ----
            json_data:
                dictionary containing all information extracted from the strain data
            demographics:
                optional demographic data of the patient
        '''
        # load the data that is available from the json_data dump
        self.load_json(json_data)
        if demographics is not None:
            self.demographics = demographics
        # compute the values of the remaining class attributes
        self.setup()

    def setup(self) -> None:
        self.cycle_length_ms =  1/self.bpm * 60 * 1000
        self.time_normalise_data()
        self.means_stds = {}
        self.time_norm_means_stds = {}

    def load_json(self, json_data: dict) -> None:
        if len(json_data['data'].keys()) > 0:
            self.data = json_data['data'] 
        else: 
            raise ValueError('No data in the json.')

        self.location = json_data['location']
        self.bpm = json_data['bpm']
        self.end_diastole_ms = json_data['end_diastole_ms']
        self.end_systole_ms = json_data['end_systole_ms']
        
        self._change_data_storage_type_to(pd.DataFrame)
    
    def time_normalise_data(self) -> None:
        self.time_normalised_data = {quantity: [] for quantity in self.data.keys()}

        for quantity in self.data.keys():
            time = self.data[quantity].iloc[:, -1]
            recorded_cycles = time.iloc[-1] / self.cycle_length_ms
            # the number of cycles to be extracted (including a possibly half 
            # finished cycle at the end of the recording)
            n_cycles = int(recorded_cycles) + (recorded_cycles - int(recorded_cycles) > PREC)

            # scale time to [0, 1] across one cardiac cycle
            scaled_time = time / self.cycle_length_ms

            for cycle in range(n_cycles):
                one_cycle = (scaled_time >= cycle + 0.0) & (scaled_time < cycle + 1.0)
                temp = copy.deepcopy(self.data[quantity].loc[one_cycle])
                temp.iloc[:, -1] /= self.cycle_length_ms
                self.time_normalised_data[quantity].append(temp)

    def normalise_time_normalised_data(self, quantity, cycle) -> None:
        mean, std = None, None
        if quantity in self.time_norm_means_stds.keys() and cycle in self.time_norm_means_stds[quantity].keys():
            mean, std = self.time_norm_means_stds[quantity][cycle]
        
        else:
            if cycle >= len(self.time_normalised_data[quantity]):
                raise ValueError(f'There are only {len(self.time_normalised_data[quantity])} < {cycle+1} recorded cardiac cycles.')

            # compute mean, std over all segments except for time
            mean = self.time_normalised_data[quantity][cycle].iloc[:, :-1].mean(axis=1)
            std = self.time_normalised_data[quantity][cycle].iloc[:, :-1].std(axis=1, ddof=1)

            if not quantity in self.time_norm_means_stds.keys():
                self.time_norm_means_stds[quantity] = {}
            self.time_norm_means_stds[quantity][cycle] = (mean, std)

        return self.time_normalised_data[quantity][cycle].iloc[:, :-1].sub(mean, axis=0).div(std + PREC, axis=0),\
            self.time_normalised_data[quantity][cycle].iloc[:, -1]
    
    def normalise_data(self, quantity) -> None:
        mean, std = None, None
        if quantity in self.means_stds.keys():
            mean, std = self.means_stds[quantity]

        else:
            # compute mean, std over all segments except for time
            mean = self.data[quantity].iloc[:, :-1].mean(axis=1)
            std = self.data[quantity].iloc[:, :-1].std(axis=1, ddof=1)

            self.means_stds[quantity] = (mean, std)

        return self.data[quantity].iloc[:, :-1].sub(mean, axis=0).div(std + PREC, axis=0),\
            self.data[quantity].iloc[:, -1]


    def _change_data_storage_type_to(self, storage_type: T):
        for quantity in self.data.keys():
            current_type = type(self.data[quantity])
            if storage_type == dict and current_type == pd.DataFrame:
                self.data[quantity] = self.data[quantity].to_dict()
            elif storage_type == pd.DataFrame and current_type == dict:
                self.data[quantity] = pd.DataFrame(self.data[quantity])
            elif storage_type == current_type:
                pass
            else:
                raise ValueError(f'Specified conversion is not possible: {str(current_type)} -> {storage_type}')


class EchocardiographyDataCollection():

    def __init__(self,
                 overwrite: bool = False,
                 data: dict = None,
                 path_to_control: str = PATH_TO_CONTROL,
                 path_to_disease: str = PATH_TO_DISEASE,
                 path_to_demographic_control: str = PATH_TO_DEMOGRAPHIC_CONTROL,
                 path_to_demographic_disease: str = PATH_TO_DEMOGRAPHIC_DISEASE,
                 path_to_cache: str = PATH_TO_CACHE
                 ) -> None:
        '''
            a class for storing an managing instances of EchocardiographyData classes
           ----
            overwrite:
                whether the data should be generated again without relying on 
                previously stored cached data; this will also update the cached data
            data:
                optional dictionary in the same format as the cached data; if not 
                None, then this is taken as the source for the data
            path_to_control:
                path to the strain data of the control group 
            path_to_disease:
                path to the strain data of the disease group
            path_to_demographic_control:
                path to the demographic- and clinical data of the control group 
            path_to_demographic_disease:
                path to the demographic- and clinical data of the disease group 
            path_to_cache:
                path to the cached data
        '''
        self.collection = None

        if data is None:
            # if the cache directory does not yet exist, create it
            if not os.path.isdir(path_to_cache):
                os.mkdir(path_to_cache)

            # create the processed_data.json file if it doesn't already exist
            if 'processed_data.json' not in os.listdir(path_to_cache) or overwrite:
                convert_echocardiography_xlsx_to_json(path_to_control, 
                                                      path_to_disease, 
                                                      path_to_cache + 'processed_data.json'
                                                      )

            # create the processed_data.json file if it doesn't already exist
            if 'demographic_data.json' not in os.listdir(path_to_cache) or overwrite:
                convert_demography_data_to_json(path_to_demographic_control, 
                                                path_to_demographic_disease, 
                                                path_to_cache + 'demographic_data.json'
                                                )
            
            # load the data from json
            self.collection = self.load_json(path=path_to_cache + 'processed_data.json',
                                             path_to_demographics=path_to_cache + 'demographic_data.json'
                                            )
        
        else:
            self.collection = data

        # store the keys as tuples that are needed to reach the patient level of the dictionary
        # also store the indices (i) and names (name) of the segments as tuples (i, name)
        self.keys_to_patients = []
        self.segments = []

        for group in self.collection.keys():
            for patient in self.collection[group].keys():
                self.keys_to_patients.append((group, patient))
                
                n_segments_old = len(self.segments)
                data = self.collection[group][patient].data
                new_segments = data[list(data.keys())[0]].columns[:-1]
                self.segments = list(set(self.segments) | set([(i, name) for i, name in enumerate(new_segments)]))

                if n_segments_old > 0 and n_segments_old != len(self.segments):
                    raise RuntimeError('At least one individuals has different segments.')

        # change the storage type back to pd.DataFrame if necessary
        self._change_data_storage_type_to(pd.DataFrame)

        self.means_stds = {}


    def load_json(self, path: str, path_to_demographics: str) -> None:
        with open(path, 'r') as in_file, open(path_to_demographics, 'r') as in_file_demographics:
            full_data = json.load(in_file)
            demographic_data = json.load(in_file_demographics)
        
            json_collection = {}

            for group in full_data.keys():
                json_collection[group] = {}
                for patient in full_data[group].keys():
                    ECD = EchocardiographyData(full_data[group][patient], 
                                               demographics=demographic_data[group][patient]
                                              )

                    json_collection[group][patient] = ECD
            
            return json_collection

    def _change_data_storage_type_to(self, storage_type) -> None:
        for group, patient in self.keys_to_patients:
                self.collection[group][patient]._change_data_storage_type_to(storage_type)

    def train_val_split(self, 
                        validation_fraction: None | float = None,
                        seed=25
                        ) -> t.Self:
        '''
            modifies self in place to be the training set and returns another 
            collection class with the validation data
        '''

        validation_collection = {}

        # division into train and validation set
        if type(validation_fraction) == float:
            # stratification for condition and location:
            indices_of_groups = {}
            for i, (group, patient) in enumerate(self.keys_to_patients):
                ECD = self.collection[group][patient]

                key = group + ' ' + ECD.location
                if not key in indices_of_groups.keys():
                    indices_of_groups[group + ' ' + ECD.location] = []

                indices_of_groups[group + ' ' + ECD.location].append(i)

            # set the randomness for the division into train and validation set
            random.seed(seed)

            for indices in indices_of_groups.values():
                random.shuffle(indices)
                n = len(indices)
                n_val = int(n * validation_fraction)

                # add the first n_val patients in the shuffled list to the 
                # validation set and delete them from the training set
                for i in indices[:n_val]:
                    group, patient = self.keys_to_patients[i]

                    if not group in validation_collection.keys():
                        validation_collection[group] = {}
                    
                    validation_collection[group][patient] = self.collection[group][patient]
                    self.collection[group].pop(patient)
            
            # update the key lists
            self.keys_to_patients = [(group, patient) for group in self.collection.keys()\
                for patient in self.collection[group].keys()]
            
            # initialise an instance of EchocardiographyDataCollection with the validation data
            validation_collection = EchocardiographyDataCollection(data=validation_collection)

            assert set(self.keys_to_patients) & set(validation_collection.keys_to_patients) == set()
                    
        elif validation_fraction is not None:
            raise TypeError('`validation_fraction` has to be either `None` or `float`.') 

        return validation_collection

    def filter_keys_for_condition(self, 
                                  condition: t.Callable[[EchocardiographyData], bool]
                                  ) -> list:
        filtered_keys = []

        for group, patient in self.keys_to_patients:
            if condition(self.collection[group][patient]):
                filtered_keys.append((group, patient))
        
        return filtered_keys

    def transform_data_to_common_grid(self, 
                                      interpolation: str = 'linear', 
                                      n_time_points: int = 100,
                                      **kwargs
                                      ) -> t.Self:
        '''
            transforms all strain data
        '''
        
        sampled_collection = copy.deepcopy(self)

        for group, patient in sampled_collection.keys_to_patients:
            ECD = sampled_collection.collection[group][patient]

            for quantity in ECD.data.keys():
                col_names = ECD.data[quantity].columns
                time = ECD.data[quantity].iloc[:, -1]

                # define the time grid for the non-time normalised data
                common_grid = np.linspace(time.iloc[0], time.iloc[-1], n_time_points)

                # define the time grids for the time normalised data
                scaled_time = time / ECD.cycle_length_ms
                recorded_cycles = scaled_time.iloc[-1]
                n_cycles = int(recorded_cycles) + (recorded_cycles - int(recorded_cycles) > PREC)
                common_grids_rel_time = []

                for cycle in range(n_cycles):
                    end_time = 1.0 + cycle
                    if cycle >= n_cycles-1:
                        end_time = scaled_time.iloc[-1]
                    common_grids_rel_time.append(np.linspace(0.0 + cycle, end_time, n_time_points))

                # initialise the dictionaries to save the interpolated data
                new_dict = {col: [] for col in col_names}
                new_norm_dict = [{col: [] for col in col_names} for _ in range(len(common_grids_rel_time))]

                # save new time grids
                new_dict[col_names[-1]] = list(common_grid)
                for i in range(n_cycles):
                    new_norm_dict[i][col_names[-1]] = list(common_grids_rel_time[i])

                norm_data = ECD.time_normalised_data[quantity]

                for segment in col_names[:-1]:
                    seg_data = ECD.data[quantity].loc[:, segment]

                    # interpolation for the non-time normalised data
                    if interpolation == 'linear' or len(time) < 5:
                        new_dict[segment] = np.interp(common_grid, time, seg_data)

                    elif interpolation == 'smoothing splines':
                        smooth_spline = make_smoothing_spline(time, seg_data, **kwargs)
                        new_dict[segment] = smooth_spline(common_grid)
                    
                    else: 
                        raise ValueError(f'{interpolation} is not a valid interpolation method.')
                    
                    # interpolation for the time normalised data
                    for i in range(n_cycles):
                        scaled_time_cycle = norm_data[i].iloc[:, -1]
                        seg_norm_data = norm_data[i].loc[:, segment]

                        if interpolation == 'linear' or len(scaled_time_cycle) < 5:
                            new_norm_dict[i][segment] = np.interp(common_grids_rel_time[i], scaled_time_cycle, seg_norm_data)
                        
                        elif interpolation == 'smoothing splines':
                            norm_smooth_spline = make_smoothing_spline(scaled_time_cycle, seg_norm_data, **kwargs)
                            new_norm_dict[i][segment] = norm_smooth_spline(common_grids_rel_time[i])
                        
                        else: 
                            raise ValueError(f'{interpolation} is not a valid interpolation method.')
                
                sampled_collection.collection[group][patient].data[quantity] = pd.DataFrame(new_dict)
                for i in range(n_cycles):
                    sampled_collection.collection[group][patient].time_normalised_data[quantity][i] = pd.DataFrame(new_norm_dict[i])

        return sampled_collection

    def summary(self, 
                C_val: 'EchocardiographyDataCollection', 
                savefig: None | str = None
                ):
        '''
            a function summarising some of the information given in the training- 
            and validation dataset
           ----
            C_val:
                an instance of EchocardiographyDataCollection containing the 
                data of the validation dataset
            savefig:
                if not None, specifies the location at which the summary figure 
                should be stored

        '''
        stats = {'sample size': 0,
                 'recorded periods': [],
                 'bpm': [],
                 'recorded times': [],
                 'mean sampling rates': [],
                 'min sampling rates': [],
                 'max sampling rates': []
        }

        summary_stats = {group: {} for group in self.collection.keys()}

        for mode, C, keys in zip(('train', 'val'), 
                                 (self.collection, C_val.collection), 
                                 (self.keys_to_patients, C_val.keys_to_patients)
                                 ):
            for group, patient in keys:
                ECD = C[group][patient]
                location = ECD.location
                
                if not ECD.location in summary_stats[group].keys():
                    summary_stats[group][location] = {'train': copy.deepcopy(stats), 'val': copy.deepcopy(stats)}

                summary_stats[group][location][mode]['sample size'] += 1
                summary_stats[group][location][mode]['bpm'].append(ECD.bpm)

                time = ECD.data[list(ECD.data.keys())[0]].iloc[:, -1]
                summary_stats[group][location][mode]['recorded times'].append(len(time))

                sampling_periods = 1/np.diff(time) * 1000 
                summary_stats[group][location][mode]['mean sampling rates'].append(np.mean(sampling_periods))
                summary_stats[group][location][mode]['min sampling rates'].append(np.min(sampling_periods))
                summary_stats[group][location][mode]['max sampling rates'].append(np.max(sampling_periods))

                quantities = list(ECD.data.keys())
                if len(quantities) > 0:
                    summary_stats[group][location][mode]['recorded periods'].append(
                        ECD.data[quantities[0]].iloc[-1, -1] / ECD.cycle_length_ms
                        )
        
        total_mean_sampling_rate = []
        total_mean_recorded_times = []

        for group in summary_stats.keys():
            group_total = 0
            group_total_more_than_one_cycle = 0
            print(f'{group}:')
            for location in summary_stats[group].keys():
                n_individuals = summary_stats[group][location]['train']['sample size']\
                    + summary_stats[group][location]['val']['sample size']
                group_total += n_individuals
                cutoff_number = 1.95
                n_more_than_one_cycle = np.sum(np.array(summary_stats[group][location]['train']['recorded periods']) > cutoff_number)\
                    + np.sum(np.array(summary_stats[group][location]['val']['recorded periods']) > cutoff_number)
                group_total_more_than_one_cycle += n_more_than_one_cycle
                print(f'    {location}: {n_individuals} individuals ({n_more_than_one_cycle/n_individuals*100:.1f}% with > {cutoff_number} cardiac cycles)')
                
                n_less_than_one_cycle = np.sum(np.array(summary_stats[group][location]['train']['recorded periods']) < 0.95)\
                    + np.sum(np.array(summary_stats[group][location]['val']['recorded periods']) < 0.95)
                print('    ' + (len(location)+2)*' ' + f'{n_less_than_one_cycle} individuals with < 0.95 recorded cycles')

                train_val_recorded_times = summary_stats[group][location]['train']['recorded times']\
                    + summary_stats[group][location]['val']['recorded times']
                mean_recorded_times = np.mean(train_val_recorded_times)
                total_mean_recorded_times += train_val_recorded_times
                std_recorded_times = np.std(train_val_recorded_times, ddof=1)
                print('    ' + (len(location)+2)*' ' + f'recorded time points: {mean_recorded_times:.3f} +- {std_recorded_times:.3f}')

                train_val_sampling_rates = summary_stats[group][location]['train']['mean sampling rates']\
                    + summary_stats[group][location]['val']['mean sampling rates']
                mean_sampling_rates = np.mean(train_val_sampling_rates)
                total_mean_sampling_rate += train_val_sampling_rates
                std_sampling_rates = np.std(train_val_sampling_rates, ddof=1)
                print('    ' + (len(location)+2)*' ' + f'mean sampling rates in [Hz]: {mean_sampling_rates:.3f} +- {std_sampling_rates:.3f}')

                max_sampling_rates = np.max(summary_stats[group][location]['train']['max sampling rates']
                    + summary_stats[group][location]['val']['max sampling rates'])
                min_sampling_rates = np.min(summary_stats[group][location]['train']['min sampling rates']
                    + summary_stats[group][location]['val']['min sampling rates'])
                print('    ' + (len(location)+2)*' ' + f'min/max sampling rates [Hz]: {min_sampling_rates:.3f}/{max_sampling_rates:.3f}')

            print(f'\n-> total: {group_total} individuals ({group_total_more_than_one_cycle/group_total*100:.1f}% with > {cutoff_number} cardiac cycles)\n')
        print(f'-> avg. recorded times: {np.mean(total_mean_recorded_times):.3f}')
        print(f'-> avg. sampling rates [Hz]: {np.mean(total_mean_sampling_rate):.3f}')

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=200)
        
        # bar chart
        xs0 = [group + '\n(' + location + ')' for group in summary_stats for location in summary_stats[group]]
        ys0_train = [summary_stats[group][location]['train']['sample size'] for group in summary_stats for location in summary_stats[group]]
        ys0_val = []
        if len(C_val.keys_to_patients) > 0:
            ys0_val = [summary_stats[group][location]['val']['sample size'] for group in summary_stats for location in summary_stats[group]]

        # scatterplot for the mean sampling rates
        xlabels1 = []
        xs1_train, xs1_val = [], []
        ys1_train, ys1_val = [], []
        
        count = 0
        for group in summary_stats:
            for location in summary_stats[group]:
                ys1_train += summary_stats[group][location]['train']['mean sampling rates']
                xs1_train += len(summary_stats[group][location]['train']['mean sampling rates']) * [count]

                if len(C_val.keys_to_patients) > 0:
                    ys1_val += summary_stats[group][location]['val']['mean sampling rates']
                    xs1_val += len(summary_stats[group][location]['val']['mean sampling rates']) * [count]

                xlabels1.append(group + '\n(' + location + ')')
                count += 1

        xs1_train, xs1_val = np.array(xs1_train), np.array(xs1_val)
        ys1_train, ys1_val = np.array(ys1_train), np.array(ys1_val)

        # scatterplot for the number of recorded periods
        xlabels2 = []
        xs2_train, xs2_val = [], []
        ys2_train, ys2_val = [], []
        
        count = 0
        for group in summary_stats:
            for location in summary_stats[group]:
                ys2_train += summary_stats[group][location]['train']['recorded periods']
                xs2_train += len(summary_stats[group][location]['train']['recorded periods']) * [count]

                if len(C_val.keys_to_patients) > 0:
                    ys2_val += summary_stats[group][location]['val']['recorded periods']
                    xs2_val += len(summary_stats[group][location]['val']['recorded periods']) * [count]
                
                xlabels2.append(group + '\n(' + location + ')')
                count += 1

        xs2_train, xs2_val = np.array(xs2_train), np.array(xs2_val)
        ys2_train, ys2_val = np.array(ys2_train), np.array(ys2_val)

        mask_below_one_cycle_train = ys2_train < 1.0
        mask_below_one_cycle_val= ys2_val < 1.0
        masks_groups_train = [xs2_train == i for i in range(len(xlabels2))]
        masks_groups_val = [xs2_val == i for i in range(len(xlabels2))]

        # plotting
        shift = 0.07
        apply_shift = len(C_val.keys_to_patients) > 0

        n_total = len(ys0_train) + len(ys0_val)
        axs[0].bar(xs0, ys0_train, label='training data', color='darkgray', edgecolor='black')
        if len(C_val.keys_to_patients) > 0: 
            axs[0].bar(xs0, ys0_val, bottom=ys0_train, label='validation data', color='darkgray', hatch='xxx', edgecolor='black')
            axs[0].legend()
        axs[0].set_title('Distribution of individuals')
 
        axs[1].scatter(xs1_train - shift*apply_shift, ys1_train, alpha=0.5, label='training data', color='black', marker='o')
        if len(C_val.keys_to_patients) > 0:
            axs[1].scatter(xs1_val+shift, ys1_val, alpha=0.5, label='validation data', color='black', marker='x') 
            axs[1].legend()
        else:
            axs[1].scatter(xs1_train, ys1_train, alpha=0.5, label='training data', color='black', marker='o')
        axs[1].set_xticks(list(range(len(xlabels1))), xlabels1)
        axs[1].set_ylabel('Sampling rate [Hz]')
        axs[1].set_title('Average sampling rates')

        axs[2].scatter(xs2_train[mask_below_one_cycle_train] - shift*apply_shift, 
                       ys2_train[mask_below_one_cycle_train], 
                       label='training data < 1.0',
                       alpha=0.5, 
                       color='tab:orange', 
                       marker='o') 
        axs[2].scatter(xs2_train[~mask_below_one_cycle_train] - shift*apply_shift, 
                       ys2_train[~mask_below_one_cycle_train], 
                       label=r'training data $\geq 1.0$',
                       alpha=0.5, 
                       color='tab:green', 
                       marker='o')
        
        for i, (mask_train, mask_val) in enumerate(zip(masks_groups_train, masks_groups_val)):
            n_train = np.sum(mask_train & mask_below_one_cycle_train)
            n_val = np.sum(mask_val & mask_below_one_cycle_val)
            axs[2].text(i-0.12, 0.8, n_train, ha='right', color='tab:orange')
            axs[2].text(i+0.12, 0.8, n_val, ha='left', color='tab:orange')

        if len(C_val.keys_to_patients) > 0:
            axs[2].scatter(xs2_val[mask_below_one_cycle_val] + shift, 
                           ys2_val[mask_below_one_cycle_val], 
                           label='validation data < 1.0',
                           alpha=0.5, 
                           color='tab:orange', 
                           marker='x') 
            axs[2].scatter(xs2_val[~mask_below_one_cycle_val] + shift, 
                           ys2_val[~mask_below_one_cycle_val], 
                           label=r'validation data $\geq 1.0$',
                           alpha=0.5, 
                           color='tab:green', 
                           marker='x') 
            axs[2].legend()
        axs[2].set_xticks(list(range(len(xlabels2))), xlabels2)
        axs[2].set_title('Number of cardiac cycles with available data')

        fig.tight_layout()

        if savefig is not None:
            plt.savefig(savefig)


    '''
    def plot_representatives(self, 
                             criterion: None | t.Callable = None,
                             data_function: None | t.Callable = None,
                             n_cols = 6,
                             dimensions = (10/6, 5/6),
                             dpi = 100,
                             group_by_type = False,
                             bbox_to_anchor = (3.2, 1.5),
                             title=None,
                             suptitle_y=0.8,
                             savefigs=None
                             ) -> None:

        if criterion is None:
            criterion = lambda ECD: True

        if data_function is None:
            # data(ECD) == (time, [X_1, X_2, ...])
            def fun(ECD):
                ls_data = ECD.time_normalised_data[0]['Longitudinal Strain-Endo']
                time = ls_data.iloc[:, -1]
                ls_segments = [{'data': ls_data.iloc[:, i], 'label': ls_data.columns[i]} for i in range(len(ls_data.columns)-1)]
                return (time, ls_segments, 'vanilla')

            data_function = fun
        
        ordered_patient = {group: {'vanilla': list(self.collection[group].keys())} for group in self.collection.keys()}
        
        if group_by_type:
            ordered_patient = {group: {} for group in self.collection.keys()}
            for group, patient in self.keys_to_patients:
                ECD = self.collection[group][patient]
                (_, _, type_label) = data_function(ECD)

                if type_label not in ordered_patient[group].keys():
                    ordered_patient[group][type_label] = []
                
                ordered_patient[group][type_label].append(patient)
        
        max_y, min_y = 0, 0
        for group, patient in self.keys_to_patients:
            ECD = self.collection[group][patient]
            if not criterion(ECD):
                continue

            (_, data, _) = data_function(ECD)
            max_y = max([max_y] + [np.max(data[i]['data']) for i in range(len(data))])
            min_y = min([min_y] + [np.min(data[i]['data']) for i in range(len(data))])

        for group in self.collection.keys():
            
            for order in sorted(ordered_patient[group]):
                n_plots = len(ordered_patient[group][order])
                n_rows = (n_plots-1)//n_cols + 1
                
                figsize_x = dimensions[0]*n_cols
                figsize_y = dimensions[1]*n_cols*(n_plots//n_cols + 1)
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(figsize_x, figsize_y), dpi=dpi)
                
                suptitle = f'{group} ({order})'
                if title is not None:
                    suptitle += f': {title}'

                print(suptitle)

                for i, patient in enumerate(ordered_patient[group][order]):
                    show_legend = False

                    ax = None
                    if n_rows <= 1:
                        ax = axs[i]
                    else:
                        row = i//n_cols
                        col = i%n_cols
                        ax = axs[row, col]

                    ECD = self.collection[group][patient]

                    if not criterion(ECD):
                        continue

                    (time, data, _) = data_function(ECD)

                    for item in data:
                        ax.plot(time, 
                                item['data'], 
                                label = item['label'] if 'label' in item.keys() else None,
                                color = item['color'] if 'color' in item.keys() else None,
                                ls = item['ls'] if 'ls' in item.keys() else None
                                )

                        if 'label' in item.keys() and i == 0:
                            show_legend = True

                    ax.set_title(group + '\n('+ patient + ')')
                    ax.set_ylim([min_y, max_y])
                    if show_legend:
                        ax.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
                                  fancybox=True, ncol=6)

                fig.tight_layout()
                plt.show()

                if savefigs is not None:
                    fig.savefig(savefigs + group + '_' + ''.join(order.split(' ')) + '.png')
    '''

def add_demographics_to_features(C: EchocardiographyDataCollection,
                                features: dict,
                                demographics_strings: list[tuple[str, str, None | dict | t.Callable]]
                            ) -> None:
    '''
        adds the demographic and clinical data to the existing patient features;
        the features dictionary is modified in place
        ----
        C:
            the EchocardiographyDataCollection instance that contains all data
        features:
            the features extracted from the data in C
        demographics_strings:
            contains a list of tuples of the form (d_type, d_name, v), where 
            d_type specifies a general group of variables, d_name the exact name
            of the variable, and if v is not None, then it can be either a 
            dictionary that is able to convert the values of the variable into 
            scalar values, or a callable that does the same thing
    '''
    available_keys_to_patients = copy.deepcopy(C.keys_to_patients)

    for variable_group_name, s, dtype in demographics_strings:
        features[s] = {}
        for (group, patient) in C.keys_to_patients:

            ECD = C.collection[group][patient]
            # if the desired feature is not available for the given individual, then
            # delete the corresponding keys from the list of available ones
            if s not in ECD.demographics[variable_group_name].keys():
                if (group, patient) in available_keys_to_patients:
                    idx = available_keys_to_patients.index((group, patient))
                    available_keys_to_patients.pop(idx)
                continue
            
            # otherwise add the information to the features
            if group not in features[s].keys():
                features[s][group] = {}

            features[s][group][patient] = ECD.demographics[variable_group_name][s]

            if isinstance(dtype, dict):
                features[s][group][patient] = dtype[features[s][group][patient]]
            elif isinstance(dtype, t.Callable):
                features[s][group][patient] = dtype(features[s][group][patient])

    # now delete all entries for individuals where there is incomplete information
    for f_name in features.keys():
        for group in list(features[f_name].keys()):
            for patient in list(features[f_name][group].keys()):
                if (group, patient) not in available_keys_to_patients:
                    features[f_name][group].pop(patient)
            if len(list(features[f_name][group].keys())) == 0:
                features[f_name].pop(group)