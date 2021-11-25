import os
import pathlib
import numpy as np
import pickle

from config.load_config import get_config
from config.config_utils import initialize_config_preproc
from preproc.preprocess import data_loading


# Initialize the configuration
cfg = get_config(filename=pathlib.Path(os.getcwd()) / 'config' / 'default_config.yml')
cfg.user = 'jyao'

# specify the loading mode: 'csv' vs 'folder'
# if csv, then loading based on a csv file
# if folder, then loading based on existing folder structure
cfg.load_mode = 'csv'
cfg.overwrite = True
cfg = initialize_config_preproc(cfg)

vec_idx_patient = [1, 310]
X, _ = data_loading(vec_idx_patient, cfg)

f_data_handle = "preproc_data_{}_{}.pkl".format(vec_idx_patient[0], vec_idx_patient[1])
f_cfg_handle = "preproc_cfg_{}_{}.pkl".format(vec_idx_patient[0], vec_idx_patient[1])

# now actually save the data
f_data = cfg.d_preproc / f_data_handle
if not f_data.exists() or cfg.overwrite:
    with open(str(f_data), 'wb') as handle:
        pickle.dump(X, handle)

f_cfg = cfg.d_preproc / f_cfg_handle
if not f_cfg.exists() or cfg.overwrite:
    with open(str(f_cfg), 'wb') as handle:
        pickle.dump(cfg, handle)

