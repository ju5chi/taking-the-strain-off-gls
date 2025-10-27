import numpy as np

from .data import EchocardiographyData, EchocardiographyDataCollection


MAST_TYPE_COLORS = {'Type-I': 'cornflowerblue',
                    'Type-II': 'gold',
                    'Type-III': 'deeppink'
                    }

def classify_deformation_pattern(ECD: EchocardiographyData, 
                                 segment: int = 0, 
                                 verbose: bool = False
                                 ) -> str:
    '''
        deformation pattern classification scheme implemented as described in the article:
        
            Thomas P. Mast et al. “Right Ventricular Imaging and Computer Simulation 
            for Electromechanical Substrate Characterization in Arrhythmogenic Right 
            Ventricular Cardiomyopathy”. In: Journal of the American College of 
            Cardiology 68.20 (2016), pp. 2185–2197. issn: 0735-1097. 
            doi: https://doi.org/10.1016/j.jacc.2016.08.061
       ----
        ECD: 
            Echocardiography data class containing the patients data
        segment:
            the segment on which the types should be computed; default is set as
            the article to be the free-wall basal segment
       ----
        Returns:
            the tye of deformation pattern
    '''
    score = 0
    data = ECD.data['Longitudinal Strain-Endo']
    time = data.iloc[:, -1].to_numpy()
    curve = data.iloc[:, segment].to_numpy()

    one_cycle = (time >= 0) & (time <= ECD.cycle_length_ms)
    time = time[one_cycle]
    curve = curve[one_cycle]

    # time to onset shortening > 90ms (this is not quite correct; it should be 
    # measured between onset-QRS and onset-shortening)
    onset_shortening = np.argmax(curve[time + ECD.end_diastole_ms <= ECD.end_systole_ms])
    time_to_onset_shortening = time[onset_shortening]

    if time_to_onset_shortening > 90.0:
        score += 1
        if verbose:
            print(f'time to onset shortening = {time_to_onset_shortening:.1f} > 90.0')
    
    # post-systolic index
    systole = time <= ECD.end_systole_ms - ECD.end_diastole_ms
    systolic_peak_strain = np.min(curve[systole])
    peak_strain = np.min(curve)
    post_systolic_index = (peak_strain - systolic_peak_strain)/peak_strain * 100

    if post_systolic_index > 10.0:
        score += 1
        if verbose:
            print(f'post-systolic index = {post_systolic_index:.1f}% > 10.0%')
    
    # systolic peak strain > -20.0
    if systolic_peak_strain > -20.0:
        score += 1
        if verbose:
            print(f'systolic peak strain = {systolic_peak_strain:.1f} > -20.0')
    
    # systolic peak strain > -10.0
    if systolic_peak_strain > -10.0:
        score += 3
        if verbose:
            print(f'systolic peak strain = {systolic_peak_strain:.1f} > -10.0')
    
    # return the score:
    if score <= 1:
        return 'Type-I'

    if score <= 3:
        return 'Type-II'
    
    return 'Type-III'


def classify_collection(C: EchocardiographyDataCollection) -> dict:
    '''
        classifies an entire EchocardiographyDataCollection and returns the results
    '''
    dpc = {}

    for group, patient in C.keys_to_patients:
        if not group in dpc.keys():
            dpc[group] = {}

        ECD = C.collection[group][patient]
        dpc[group][patient] = classify_deformation_pattern(ECD)
    
    return dpc