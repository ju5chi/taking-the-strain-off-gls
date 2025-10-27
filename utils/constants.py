import os

absolute_path_to_cwd = os.getcwd()

# path variables for source of data, destination of results, and storage of 
# intermediate results
PATH_TO_CONTROL = absolute_path_to_cwd + '/data/Control/'
PATH_TO_DISEASE = absolute_path_to_cwd + '/data/Disease/'
PATH_TO_DEMOGRAPHIC_CONTROL = absolute_path_to_cwd + '/data/demographics/control/'
PATH_TO_DEMOGRAPHIC_DISEASE = absolute_path_to_cwd + '/data/demographics/disease/'
PATH_TO_CACHE = absolute_path_to_cwd + '/data/cache/'
PATH_TO_RESULTS = absolute_path_to_cwd + '/data/results/'

# the floating point precision below which two numbers should be considered equal
PREC = 1e-10

# contains information about the location of the data in the xlsx sheets:
# first level of the dictionary contains the sheet names as keys and the second 
# level of the dictionary contains all information needed to extract the desired 
# value
DATA_LOCATION = {
    'Strain-Endo': 
        {'segments': [[0, 5]], 'time': 10, 'prefix': ['Longitudinal']}, 
    'Strain Rate-Endo': 
        {'segments': [[0, 5]], 'time': 10, 'prefix': ['Longitudinal']}
    }

# some general information about the patient and their recording in the strain files:
# first level of the dictionary contains the sheet names as keys, the second level 
# of the dictionary contains the datatype, and the third level all information needed
# to extract the desired value
META_DATA_LOCATION = {
    'Data':{
        'dates': {
            'date_of_evaluation': {'location': (1, 'Unnamed: 1'), 're': r'\'?(\d{4})-(\d{2})-(\d{2})', 'type': None}
        },
        'continuous': {
            'bpm': {'location': (15, 'Unnamed: 1'), 're': None, 'type': float}, 
            'end_diastole_ms': {'location': (17, 'Unnamed: 4'), 're': '(\d+).*ms', 'type': float}, 
            'end_systole_ms': {'location': (18, 'Unnamed: 4'), 're': '(\d+).*ms', 'type': float} 
        }
    }
}

# demographic and clinical data:
# first level of the dictionary contains the datatype, and the third level all 
# information needed to extract the desired value
DEMOGRAPHICS_DATA_LOCATION = {
    'dates': {
        'Date of birth': {'format': r'(\d{2})\/(\d{2})\/(\d{4})'},
        'Date of diagnosis (Definite ARVC)': {'format': r'(\d{2})\/(\d{2})\/(\d{4})'},
        'Genetic analysis report date': {'format': r'(\d{2})\/(\d{2})\/(\d{4})'},
        'Date ICD implantation': {'format': r'(\d{2})\/(\d{2})\/(\d{4})'},
        'Date TTE': {'format': r'(\d{2})\/(\d{2})\/(\d{4})'}
    },

    'continuous': {
        'Height' : {'SI factor': 0.01},
        'Weight' : {'SI factor': 1.0},
        'days_tte1_va_dea_ht': {'type': float},
        'days_tte1_dea_ht': {'type': float},
        'T wave inversion': {'type': float},
        'RVEF': {'type': float},
        'age': {'type': float}
    },

    'TFC' : {
        'Echocardiography TFC': {'levels': {'0': None, '1': 'major', '2': 'minor'}},
        'CMR TFC': {'levels': {'0': None, '1': 'major', '2': 'minor'}},
        'Echocardiography and/or CMR TFC': {'levels': {'0': None, '1': 'major', '2': 'minor'}},
        'RV angiography TFC': {'levels': {'0': None, '1': 'major', '2': 'minor'}},
        'Tissue characterisation TFC': {'levels': {'0': None, '1': 'major', '2': 'minor'}},
        'Repolarisation abnormalities TFC': {'levels': {'0': None, '1': 'major', '2': 'minor'}},
        'Depolarisation and/or conduction abnormalities TFC': {'levels': {'0': None, '1': 'major', '2': 'minor'}},
        'Arrhythmias TFC': {'levels': {'0': None, '1': 'major', '2': 'minor'}},
        'Family history TFC': {'levels': {'0': None, '1': 'major', '2': 'minor'}}
    },

    'factors': {
        'Sex': {'levels': {'1': 'male', '2': 'female'}},
        'Index patient': {'levels': {'0': 'no', '1': 'yes', '2': None}},
        'Genetic analysis': {'levels': {'0': 'no', '1': 'yes', '2': None}},
        'Gene 1 (P/LP)': {},
        'Gene 2': {},
        'Variant(s)': {},
        'Zygosity Gene 1': {},
        'Zygosity Gene 2': {},
        'Pathogenicity Gene 2': {},
        'icdatdech1': {},
        'va_event_timing_g2': {},
        'svts_event_timing_g2': {},
        'ind_tte1_va_dea_ht_event': {'type': int},  # indicator whether the 
                                                    # composite endpoint of VA + 
                                                    # death + HT is reached
        'ind_tte1_dea_ht_event': {'type': int}, # indicator whether the composite
                                                # endpoint of death + HT is reached
        'ind_tte1_va_event': {'type': int}, # indicator whether VA event reached
        'Syncope': {'levels': {'0': 'no', '1': 'yes', '2': None}},
        'Atrial Tachycardia > 30 sec (ATach30)': {'levels': {'0': 'no', '1': 'yes', '2': None}}
    }
}