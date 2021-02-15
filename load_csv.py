import pandas as pd
import numpy as np


def load_csv_params(cfg):
    pd_csv = pd.read_csv(str(cfg.d_csv / cfg.f_csv))
    pd_csv_headers = pd_csv.columns

    # get the valid columns in the CSV file to access
    # i.e. parsing the column of the patient id and OD and OS
    idx_col_all = np.arange(0, len(pd_csv_headers))
    idx_col_patient_id = idx_col_all[pd_csv_headers.str.match('Pt\n')]
    idx_col_OD_valid = idx_col_all[pd_csv_headers.str.match('OD\n')]
    idx_col_OS_valid = idx_col_all[pd_csv_headers.str.match('OS\n')]

    # get the index of the columns corresponding to the features
    if cfg.str_feature != 'disease':
        idx_col_OD_feature = idx_col_all[pd_csv_headers.str.match('OD: {}'.format(cfg.str_feature))]
        idx_col_OS_feature = idx_col_all[pd_csv_headers.str.match('OS: {}'.format(cfg.str_feature))]
    else:
        idx_col_OD_feature = idx_col_all[pd_csv_headers.str.match('OD\n')]
        idx_col_OS_feature = idx_col_all[pd_csv_headers.str.match('OS\n')]

    # now obtain all the patient IDs and necessary labels
    vec_str_patient_id = pd_csv[pd_csv_headers[idx_col_patient_id]].T.astype('int').to_numpy()[0]

    # these two variables hold the true disease label, 0 - excluded, 1 normal, 2 - Dry AMD, 3 - Wet AMD
    vec_OD_valid = pd_csv[pd_csv_headers[idx_col_OD_valid][0]].to_numpy()
    vec_OS_valid = pd_csv[pd_csv_headers[idx_col_OS_valid][0]].to_numpy()

    # obtain the correct feature labels
    vec_OD_feature = pd_csv[pd_csv_headers[idx_col_OD_feature][0]].to_numpy()
    vec_OS_feature = pd_csv[pd_csv_headers[idx_col_OS_feature][0]].to_numpy()

    # For OD/OS valid labels, test if all values in 0-3, set to 0 otherwise
    if vec_OD_valid.dtype == 'object':
        idx_invalid_OD_curr = np.logical_not(np.isin(vec_OD_valid, ['0', '1', '2', '3']))
        vec_OD_valid[idx_invalid_OD_curr] = 0
        vec_OD_valid = vec_OD_valid.astype(np.int)
    else:
        idx_invalid_OD_curr = np.logical_not(np.isin(vec_OD_valid, [0, 1, 2, 3]))
        vec_OD_valid[idx_invalid_OD_curr] = 0

    if cfg.str_feature != 'disease':
        vec_OD_feature[idx_invalid_OD_curr] = np.nan
    else:
        # correct the labels so that it's consistent with the other features when using disease label
        vec_OD_feature = vec_OD_valid.astype(np.float)
        vec_OD_feature = vec_OD_feature - 1
        vec_OD_feature[vec_OD_feature < 0] = np.nan

    if vec_OS_valid.dtype == 'object':
        idx_invalid_OS_curr = np.logical_not(np.isin(vec_OS_valid, ['0', '1', '2', '3']))
        vec_OS_valid[idx_invalid_OS_curr] = 0
        vec_OS_valid = vec_OS_valid.astype(np.int)
    else:
        idx_invalid_OS_curr = np.logical_not(np.isin(vec_OS_valid, [0, 1, 2, 3]))
        vec_OS_valid[idx_invalid_OS_curr] = 0

    if cfg.str_feature != 'disease':
        vec_OS_feature[idx_invalid_OS_curr] = np.nan
    else:
        # correct the labels so that it's consistent with the other features when using disease label
        vec_OS_feature = vec_OS_valid.astype(np.float)
        vec_OS_feature = vec_OS_feature - 1
        vec_OS_feature[vec_OS_feature < 0] = np.nan

    return vec_str_patient_id, vec_OD_feature, vec_OS_feature
