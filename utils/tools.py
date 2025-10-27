import os
import re
import typing as t
import pandas as pd
import json
import numpy as np
import datetime

from .constants import PATH_TO_RESULTS, DATA_LOCATION, META_DATA_LOCATION, \
    DEMOGRAPHICS_DATA_LOCATION

T = t.TypeVar('T')


### reading in of data

def convert_echocardiography_xlsx_to_json(path_to_control: str,
                                          path_to_disease: str,
                                          path_to_json: str) -> dict:
    '''
        Converts the raw data from xlsx format into a more concice json format.
        More precisely, it arranges the following information into the indicated 
        hierarchy:

            |- <group> (either `control` or `disease`)
                |- <location> (the origin location of the data)
                    |- <patient ID> (the ID assigned to the given patient)
                        |- data
                        |- heart rate in beats per minute (bpm)
                        |- end of the first recorded diastole (in ms)
                        |- end of the first recorded systole (in ms)
                    
       ----
        `path_to_control`:
            the path(s) to the data of the group `control`
        `path_to_disease`:
            the path(s) to the data of the group `disease`
        `path_to_json`:
            the path to the location where the json dump should be stored at
    '''

    collection = {'control': {},
                  'disease': {}
                 }

    # navigate the folders and load the xlsx files
    for group, path in zip(collection.keys(), [path_to_control, path_to_disease]):
        for location in os.listdir(path):
            path_to_location = path + location + '/'
            for individual in os.listdir(path_to_location):
                path_to_individual = path_to_location + individual + '/'
                for file_name in os.listdir(path_to_individual):

                    match_xslx = re.search('(.*)\.xlsx$', file_name)
                    if not match_xslx:
                        continue

                    # load the xlsx file
                    excel_file = pd.read_excel(path_to_individual + file_name, 
                                               sheet_name=None
                                               )
                    # get patient ID from filename and convert it to uppercase
                    patient_id = match_xslx.group(1).upper()
                    
                    # initialise a dictionary that holds all important information 
                    # and will later be saved as part of a json file
                    json_data = {
                        'data': {},
                        'location': location
                        }

                    # add important metadata items to the json
                    for sheet_name in META_DATA_LOCATION.keys():
                        for data_type, data_type_dicts in  META_DATA_LOCATION[sheet_name].items():
                            for property_name, property_info in data_type_dicts.items():
                                json_data[property_name] = excel_file[sheet_name].loc[property_info['location']]
                                
                                # extract information out of string using regex
                                if property_info['re'] is not None:
                                    if data_type == 'continuous':
                                        if type(json_data[property_name]) != str:
                                            raise ValueError(f'The value of `{property_name}` is not a string.')

                                        match = re.search(property_info['re'], json_data[property_name])
                                        if match:
                                            json_data[property_name] = match.group(1)
                                        else:
                                            raise RuntimeError(f'No float for `{property_name}` found.')

                                    elif data_type == 'dates':
                                        json_data[property_name] = extract_date_information(json_data[property_name], 
                                                                                            re_pattern=property_info['re']
                                                                                           )
                                    
                                    else:
                                        raise ValueError(f'No defined behaviour for datatype {data_type}.')

                                # cast original value to new type
                                if property_info['type'] is not None:
                                    json_data[property_name] = property_info['type'](json_data[property_name])
            
                    # extract the measurement data
                    for quantity in DATA_LOCATION.keys():
                        # resolve ambiguity of the names
                        quantity_misspelled = find_string_instance(quantity, excel_file.keys())
                        if quantity_misspelled is None:
                            raise RuntimeError(f'The quantity `{quantity}` is not present in the file.')

                        # extract the desired information in a suitable format
                        for i, locs in enumerate(DATA_LOCATION[quantity]['segments']):
                            temp = excel_file[quantity_misspelled].loc[locs[0]:locs[1]].transpose()

                            # some files may only have the time saved; in that case skip the quantity
                            if len(temp.columns) < 6:
                                 continue

                            temp.columns = temp.iloc[0] # rename the columns
                            temp = temp.drop(temp.columns.name) # remove duplicate row
                            if i > 0:
                                temp.columns.name =excel_file[quantity_misspelled].loc[locs[0]-1].iloc[0]
                            temp.index = [i for i in range(len(temp))] # reindex
                            key = DATA_LOCATION[quantity]['prefix'][i] + ' ' + quantity

                            temp['Time [ms]'] = list(
                                excel_file[quantity_misspelled]\
                                .loc[DATA_LOCATION[quantity]['time']]\
                                .iloc[1:] - json_data['end_diastole_ms']
                                )

                            temp = temp.astype('float64')

                            # convert the DF to a dictionary
                            json_data['data'][key] = temp.to_dict()

                    collection[group][patient_id] = json_data

    # dump the converted data to json
    with open(path_to_json, 'w') as out_file:
        json.dump(collection, out_file, cls=NumpyEncoder)



def extract_date_information(date: datetime.datetime | str, 
                             re_pattern: str
                             ) -> dict:
    
    # if the date is handled by excel as a datetime object:
    if isinstance(date, datetime.datetime):
        return {'year': date.year, 
                'month': date.month, 
                'day': date.day
               }

    # if the date is read as a string:
    elif isinstance(date, str):
        match = re.search(re_pattern, date)
        if match is not None:
            return {'year': int(match.group(3)), 
                    'month': int(match.group(2)), 
                    'day': int(match.group(1))
                   }

        else:
            raise RuntimeError(f'No match found for pattern {re_pattern} in string {date}.')
    
    # if the date is missing
    elif pd.isna(date):
        return {'year': None, 
                'month': None, 
                'day': None
                }

    else:
        raise RuntimeError(f'Date has an invalid format with type {type(date)}: {date}.')


### conversion of data

def convert_demography_data_to_json(path_to_control: str,
                                    path_to_disease: str,
                                    path_to_json: str) -> dict:
    '''
        Converts the raw data into a more concice json format.
        More precisely, it arranges the following information into the indicated hierarchy:

            |- <group> (either `control` or `disease`)
                |- <patient ID> (the ID assigned to the given patient)
                    |- data
                    
       ----
        `path_to_control`:
            the path(s) to the data of the group `control`
        `path_to_disease`:
            the path(s) to the data of the group `disease`
        `path_to_json`:
            the path to the location where the json dump should be stored at
    '''


    collection = {'control': {},
                  'disease': {}
                 }

    # navigate the folders and load the xlsx files
    for group, path in zip(['control', 'disease'], [path_to_control, path_to_disease]):
        for file_name in os.listdir(path):
            df = None

            match_xslx = re.search('(.*)\.xlsx$', file_name)
            match_csv = re.search('(.*)\.csv$', file_name)

            if match_xslx:
                # load the xlsx file
                excel_file = pd.read_excel(path + file_name, 
                                        sheet_name=None, 
                                        index_col=0,
                                        header=2
                                        )
                
                # crop the dataframe to everything but the first two lines
                df = excel_file['Data Collection Sheet - ARVC'].iloc[2:]

            elif match_csv:
                # load the csv file
                df = pd.read_csv(path + file_name,
                                 index_col=0,
                                 header=0
                                )

            else:
                continue


            # convert patient IDs to uppercase
            df.index = df.index.str.upper()

            # start out by initializing dictionaries for all patients
            for patient in df.index:
                if patient not in collection[group].keys():
                    collection[group][patient] = {}


                # add the date-data to the dictionaries
                date_cols = DEMOGRAPHICS_DATA_LOCATION['dates']
                if 'dates' not in collection[group][patient].keys():
                    collection[group][patient]['dates'] = {}

                for date_name, date_info in date_cols.items():
                    if date_name not in df.columns:
                        continue
                    date = df.loc[patient, date_name]

                    collection[group][patient]['dates'][date_name]\
                        = extract_date_information(date, re_pattern=date_info['format'])


                # add the continuous data to the dictionaries (after converting them to SI units)
                cont_cols = DEMOGRAPHICS_DATA_LOCATION['continuous']
                if 'continuous' not in collection[group][patient].keys():
                    collection[group][patient]['continuous'] = {}

                for cont_name, cont_info in cont_cols.items():
                    if cont_name not in df.columns:
                        continue

                    value = df.loc[patient, cont_name]

                    if 'SI factor' in cont_info.keys():
                        value *= cont_info['SI factor']

                    collection[group][patient]['continuous'][cont_name] = value


                # add the factor variables to the dictionaries
                factor_cols = DEMOGRAPHICS_DATA_LOCATION['factors']
                if 'factors' not in collection[group][patient].keys():
                    collection[group][patient]['factors'] = {}
                
                tfc_cols = DEMOGRAPHICS_DATA_LOCATION['TFC']
                if 'TFC' not in collection[group][patient].keys():
                    collection[group][patient]['TFC'] = {}

                for variable_group_name, variable_group_cols in zip(['factors', 'TFC'], [factor_cols, tfc_cols]):
                    for name, info in variable_group_cols.items():
                        if name not in df.columns:
                            continue

                        if pd.isna(df.loc[patient, name]):
                            collection[group][patient][variable_group_name][name] = None

                        # use the level information (if provided)
                        elif 'levels' in info.keys():
                            str_value = str(df.loc[patient, name])
                            value = info['levels'][str_value]
                            collection[group][patient][variable_group_name][name] = value

                        elif 'type' in info.keys():
                            value = info['type'](df.loc[patient, name])
                            collection[group][patient][variable_group_name][name] = value
                        
                        else:
                            str_value = str(df.loc[patient, name])
                            collection[group][patient][variable_group_name][name] = str_value
    
    # calculate more properties using the extracted information
    for group in collection.keys():
        for patient in collection[group].keys():
            patient_data = collection[group][patient]

            if 'dates' in patient_data.keys():
                if 'Date of birth' in patient_data['dates'].keys()\
                    and 'Date TTE' in patient_data['dates'].keys():

                    dob = patient_data['dates']['Date of birth']
                    dtte = patient_data['dates']['Date TTE']

                    age = dtte['year'] - dob['year']
                    if dtte['month'] < dob['month']:
                        age -= 1
                    elif dtte['month'] == dob['month'] and dtte['day'] < dob['day']:
                        age -= 1

                    collection[group][patient]['continuous']['age_at_tte'] = age
    
    # dump the converted data to json
    with open(path_to_json, 'w') as out_file:
        json.dump(collection, out_file, cls=NumpyEncoder)   
               

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()

        return super().default(obj)


### general tools

def find_string_instance(target_string: str, 
                         possible_strings: list[str], 
                         return_unambiguous: bool = False) -> None | str | tuple[None | str, str]:
    '''
        looks for the existance of the string target_string inside the list 
        possible_strings; for there to be a match, the target string has to coincide 
        with one of the possible strings after removal of all whitespaces, '-', 
        '(', ")" and ignoring upper-/lower case letters
       ----
        target_string:
            the string to be matched against `possible_strings`
        possible_strings:
            a list of string candidates
        return_unambiguous:
            whether an umambiguous version of the string should be returned when 
            a match has been found
       ----
        Returns:
            None if no match has been found and a string otherwise; if 
            `return_unambiguous` is True, then a tuple containing two strings is 
            returned
    '''
    simplify_string = lambda s: ''.join(filter(lambda ch: ch not in ' -()', s)).lower()

    target_instance = None
    for s in possible_strings:
        if simplify_string(target_string) == simplify_string(s):
            target_instance = s
            break
            
    if return_unambiguous:
        return (target_instance, simplify_string(target_string))
    
    return target_instance