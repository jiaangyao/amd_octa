import pandas as pd
import numpy as np


def load_csv_params(cfg, bool_mode_full=True):
    pd_csv = pd.read_csv(str(cfg.d_csv / cfg.f_csv))
    pd_csv_headers = pd_csv.columns

    # get the valid columns in the CSV file to access
    # i.e. parsing the column of the patient id and OD and OS
    idx_col_all = np.arange(0, len(pd_csv_headers))
    idx_col_patient_id = idx_col_all[pd_csv_headers.str.match('Pt\n')][0]

    # now obtain all the patient IDs and necessary labels
    vec_str_patient_id = pd_csv[pd_csv_headers[idx_col_patient_id]].T.astype('int').to_numpy()

    if bool_mode_full:
        idx_col_OD_valid = idx_col_all[pd_csv_headers.str.match('OD\n')][0]
        idx_col_OS_valid = idx_col_all[pd_csv_headers.str.match('OS\n')][0]

        # get the index of the columns corresponding to the features
        if cfg.str_feature != 'disease':
            idx_col_OD_feature = idx_col_all[pd_csv_headers.str.match('OD: {}'.format(cfg.str_feature))][0]
            idx_col_OS_feature = idx_col_all[pd_csv_headers.str.match('OS: {}'.format(cfg.str_feature))][0]
        else:
            idx_col_OD_feature = idx_col_all[pd_csv_headers.str.match('OD\n')][0]
            idx_col_OS_feature = idx_col_all[pd_csv_headers.str.match('OS\n')][0]

        # these two variables hold the true disease label, 0 - excluded, 1 normal, 2 - Dry AMD, 3 - Wet AMD
        vec_OD_valid = pd_csv[pd_csv_headers[idx_col_OD_valid]].to_numpy()
        vec_OS_valid = pd_csv[pd_csv_headers[idx_col_OS_valid]].to_numpy()

        # obtain the correct feature labels
        vec_OD_feature = pd_csv[pd_csv_headers[idx_col_OD_feature]].to_numpy()
        vec_OS_feature = pd_csv[pd_csv_headers[idx_col_OS_feature]].to_numpy()

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

        out_csv = pd_csv.copy()
        out_csv = out_csv.astype('float')
        for idx_col in range(len(pd_csv_headers)):
            if idx_col not in [idx_col_patient_id, idx_col_OD_valid, idx_col_OS_valid, idx_col_OD_feature, idx_col_OS_feature]:
                out_csv.loc[:, pd_csv_headers[idx_col]] = np.nan

        idx_row_all = np.arange(out_csv.shape[0])
        idx_row_OD_valid = np.logical_not(np.isnan(out_csv.loc[:, pd_csv_headers[idx_col_OD_feature]].to_numpy()))
        idx_row_OS_valid = np.logical_not(np.isnan(out_csv.loc[:, pd_csv_headers[idx_col_OS_feature]].to_numpy()))

        out_csv.loc[idx_row_all[idx_row_OD_valid], pd_csv_headers[idx_col_OD_feature]] = 9999
        out_csv.loc[idx_row_all[idx_row_OS_valid], pd_csv_headers[idx_col_OS_feature]] = 9999

        pd_csv = pd_csv.copy()
        out_csv = out_csv
        vec_csv_col = [idx_col_patient_id, idx_col_OD_valid, idx_col_OS_valid, idx_col_OD_feature, idx_col_OS_feature]

    else:
        vec_OD_feature = None
        vec_OS_feature = None

        pd_csv = None
        out_csv = None
        vec_csv_col = None

    return vec_str_patient_id, vec_OD_feature, vec_OS_feature, pd_csv, out_csv, vec_csv_col
