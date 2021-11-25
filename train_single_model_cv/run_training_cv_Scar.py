import os
import pathlib
import pickle

from config.load_config import get_config
from config.config_utils import initialize_config_preproc, initialize_config_split, initialize_config_training
from preproc.preprocess import generate_labels, correct_data_label
from preproc.train_val_test_split import prepare_data_for_train_cv
from train_single_model_cv.run_training_cv import run_training_cv


if __name__ == '__main__':
    # Configuring the files here for now
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

    # name of particular feature that will be used
    # note if want to test for disease label then have to specify this to be 'disease'
    # otherwise it has to be one of ['IRF/SRF', 'Scar', 'GA', 'CNV', 'Large PED']
    cfg.str_feature = 'Scar'

    # whether or not to make the training set balanced - note this will give you imbalanced test set
    cfg.balanced = False

    # for CV script then here the cross validation mode should be enabled
    cfg.cv_mode = True

    # specify model architecture and whether to use debug mode
    cfg.str_arch = 'arch_022'
    cfg.debug_mode = False

    # now load the preprocessed data and the label
    f_data_handle = "preproc_data_{}_{}.pkl".format(vec_idx_patient[0], vec_idx_patient[1])
    f_data = cfg_template.d_preproc / f_data_handle
    with open(str(f_data), 'rb') as handle:
        X = pickle.load(handle)

    y = generate_labels(cfg.str_feature, cfg, bool_append_csv_to_cfg=True)

    # now prepare data for training
    cfg = initialize_config_split(cfg)
    X, y = correct_data_label(X, y, cfg)
    vec_Xs, vec_ys = prepare_data_for_train_cv(X, y, cfg)

    # finally set the training parameters
    cfg = initialize_config_training(cfg, bool_debug=cfg.debug_mode)
    cfg = run_training_cv(vec_Xs, vec_ys, cfg)
