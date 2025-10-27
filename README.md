# Taking the Strain off GLS
## Novel Strain-Imaging Based Features for the Diagnosis of Arrhythmogenic Right Ventricular Cardiomyopathy

Accompanying codebase of the master's thesis with the same title.

Implements functionality to
- read in and process longitudinal strain data of individuals,
- extract features from these measurements,
- define classification (and survival analysis) models to explore the data,
- perform model selection on these models,
- summarise the results in tables and plots.

## Requirements:

The requirements to run the python scripts are given as a conda evironment and are specified in `env.yaml`. To set up the environment, run `conda env create -f env.yaml`.

## Reading-in New Data:

In general, the data of healthy and diseased individuals needs to be provided in different directories. The default locations are specified in `./utils/constants.py`, but can be changed. The read-in procedures to extract the raw strain data as well as demographic/clinical data are defined in `./utils/tools.py`. Parameters that are used to guide these functions are defined in `./utils/constants.py`. If the read-in functionality is to be applied to data stored in a different format, these functions and constants need to be modified accordingly. While the rest of the codebase is mostly independent of the underlying data, it should be noted that it is neither intended to use nor tested with any other format.

## Example Applications

An example application of the most important features can be found in the jupyter-notebook `model_selection_run.py`. Visualisations of the underlying data and summary information can be generated as demonstrated in `visualisations.py`.
