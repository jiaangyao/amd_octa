import copy
import os
import pathlib
import numpy as np
import pickle

from config.load_config import get_config
from config.config_utils import initialize_config_preproc
from preproc.preprocess import data_loading, generate_labels

# Initialize the configuration
cfg_template = get_config(filename=pathlib.Path(os.getcwd()).parent / 'config' / 'default_config.yml')
cfg_template.user = 'jyao'
cfg_template.load_mode = 'csv'
cfg_template.overwrite = True
cfg_template = initialize_config_preproc(cfg_template)

# now load the actual cfg generated from the data
vec_idx_patient = [1, 310]
f_cfg_handle = "preproc_cfg_{}_{}.pkl".format(vec_idx_patient[0], vec_idx_patient[1])
f_cfg = cfg_template.d_preproc / f_cfg_handle
with open(str(f_cfg), 'rb') as handle:
    cfg = pickle.load(handle)

# sanity check
y_disease = generate_labels('disease', cfg)
y_irf = generate_labels('IRF/SRF', cfg)
y_scar = generate_labels('Scar', cfg)
y_ga = generate_labels('GA', cfg)
y_cnv = generate_labels('CNV', cfg)
y_ped = generate_labels('Large PED', cfg)

# now load the previous csv file
cfg_prev = copy.deepcopy(cfg)
cfg_prev.f_csv = 'DiseaseLabelsThrough305.csv'
y_disease_prev = generate_labels('disease', cfg_prev)

# get the count of nan values
vec_str_patients = copy.deepcopy(cfg.vec_str_patients)
vec_str_patients = np.stack(vec_str_patients, axis=0)
vec_str_patients_nan = vec_str_patients[np.isnan(y_disease)]

# get the count of nan values in previous csv
vec_str_patients_prev_nan = vec_str_patients[np.isnan(y_disease_prev)]

# get the class label numbers in current
vec_class_count = []
vec_class_count_prev = []
for idx_class in range(3):
    vec_class_count.append(np.sum(y_disease == idx_class))
    vec_class_count_prev.append(np.sum(y_disease_prev == idx_class))

# now get the valid labels for features
y_disease_valid = y_disease.copy()[np.logical_not(np.isnan(y_disease))]
y_irf_valid = y_irf[np.logical_not(np.isnan(y_disease))]
y_scar_valid = y_scar[np.logical_not(np.isnan(y_disease))]
y_ga_valid = y_ga[np.logical_not(np.isnan(y_disease))]
y_cnv_valid = y_cnv[np.logical_not(np.isnan(y_disease))]
y_ped_valid = y_ped[np.logical_not(np.isnan(y_disease))]

# now get the class label numbers for all features
vec_irf_count = []
vec_scar_count = []
vec_ga_count = []
vec_cnv_count = []
vec_ped_count = []
for idx_class in range(3):
    vec_irf_count.append(np.sum(y_irf_valid == idx_class))
    vec_scar_count.append(np.sum(y_scar_valid == idx_class))
    vec_ga_count.append(np.sum(y_ga_valid == idx_class))
    vec_cnv_count.append(np.sum(y_cnv_valid == idx_class))
    vec_ped_count.append(np.sum(y_ped_valid == idx_class))

vec_irf_count.append(np.sum(np.isnan(y_irf_valid)))
vec_scar_count.append(np.sum(np.isnan(y_scar_valid)))
vec_ga_count.append(np.sum(np.isnan(y_ga_valid)))
vec_cnv_count.append(np.sum(np.isnan(y_cnv_valid)))
vec_ped_count.append(np.sum(np.isnan(y_ped_valid)))

# obtain the subjects with missing feature labels
vec_str_patients_valid = vec_str_patients[np.logical_not(np.isnan(y_disease))]
vec_str_patients_nan_irf = vec_str_patients_valid[np.isnan(y_irf_valid)]
vec_str_patients_nan_scar = vec_str_patients_valid[np.isnan(y_scar_valid)]
vec_str_patients_nan_ga = vec_str_patients_valid[np.isnan(y_ga_valid)]
vec_str_patients_nan_cnv = vec_str_patients_valid[np.isnan(y_cnv_valid)]
vec_str_patients_nan_ped = vec_str_patients_valid[np.isnan(y_ped_valid)]

# now get the class label numbers for all features full
vec_irf_count_full = []
vec_scar_count_full = []
vec_ga_count_full = []
vec_cnv_count_full = []
vec_ped_count_full = []
for idx_class in range(3):
    vec_irf_count_full.append(np.sum(y_irf == idx_class))
    vec_scar_count_full.append(np.sum(y_scar == idx_class))
    vec_ga_count_full.append(np.sum(y_ga == idx_class))
    vec_cnv_count_full.append(np.sum(y_cnv == idx_class))
    vec_ped_count_full.append(np.sum(y_ped == idx_class))

vec_irf_count_full.append(np.sum(np.isnan(y_irf)))
vec_scar_count_full.append(np.sum(np.isnan(y_scar)))
vec_ga_count_full.append(np.sum(np.isnan(y_ga)))
vec_cnv_count_full.append(np.sum(np.isnan(y_cnv)))
vec_ped_count_full.append(np.sum(np.isnan(y_ped)))


print('nothing')
