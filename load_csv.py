import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path('/Users/jyao/Downloads/')
csv_fname = 'FeatureLabeling.csv'

pd_csv = pd.read_csv(str(csv_path / csv_fname))
pd_csv_headers = pd_csv.columns

# name of particular feature that will be used
# note this has to match what's in the CSV file
str_feature = 'IRF/SRF'

# get the valid columns in the CSV file to access
idx_col_all = np.arange(0, len(pd_csv_headers))
idx_col_patient_id = idx_col_all[pd_csv_headers.str.match('Pt\n')]
idx_col_OD_valid = idx_col_all[pd_csv_headers.str.match('OD\n')]
idx_col_OS_valid = idx_col_all[pd_csv_headers.str.match('OS\n')]

# get the index of the columns corresponding to the features
idx_col_OD_feature = idx_col_all[pd_csv_headers.str.match('OD: {}'.format(str_feature))]
idx_col_OS_feature = idx_col_all[pd_csv_headers.str.match('OS: {}'.format(str_feature))]

# now obtain all the patient IDs and necessary labels
vec_patient_id = pd_csv[pd_csv_headers[idx_col_patient_id]].T.astype('int').to_numpy()
vec_OD_valid = pd_csv[pd_csv_headers[idx_col_OD_valid][0]].to_numpy()
vec_OS_valid = pd_csv[pd_csv_headers[idx_col_OS_valid][0]].to_numpy()

# obtain the correct feature labels
vec_OD_feature = pd_csv[pd_csv_headers[idx_col_OD_feature][0]].to_numpy()
vec_OS_feature = pd_csv[pd_csv_headers[idx_col_OS_feature][0]].to_numpy()

# For OD/OS valid labels, test if all values in 0-3, set to 0 otherwise
if vec_OD_valid.dtype == 'object':
    idx_invalid_OD_curr = np.logical_not(np.isin(vec_OD_valid,['0', '1', '2', '3']))
    vec_OD_valid[idx_invalid_OD_curr] = '0'
    vec_OD_valid = vec_OD_valid.astype(np.int)
else:
    idx_invalid_OD_curr = np.logical_not(np.isin(vec_OD_valid, [0, 1, 2, 3]))
    vec_OD_valid[idx_invalid_OD_curr] = 0
vec_OD_feature[idx_invalid_OD_curr] = np.nan

if vec_OS_valid.dtype == 'object':
    idx_invalid_OS_curr = np.logical_not(np.isin(vec_OS_valid, ['0', '1', '2', '3']))
    vec_OS_valid[idx_invalid_OS_curr] = '0'
    vec_OS_valid = vec_OS_valid.astype(np.int)
else:
    idx_invalid_OS_curr = np.logical_not(np.isin(vec_OS_valid, [0, 1, 2, 3]))
    vec_OS_valid[idx_invalid_OS_curr] = 0
vec_OS_feature[idx_invalid_OS_curr] = np.nan

print('nothing')