import os
import pathlib
import pickle

from config.load_config import get_config
from config.config_utils import initialize_config_preproc
from train_repeated_model_binary.run_repeated_training_binary import run_repeated_training_binary


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
        cfg_in = pickle.load(handle)
    cfg_in.vec_idx_patient = vec_idx_patient

    # name of particular feature that will be used
    # note if want to test for disease label then have to specify this to be 'disease'
    # otherwise it has to be one of ['IRF/SRF', 'Scar', 'GA', 'CNV', 'Large PED']
    cfg_in.str_feature = 'disease'

    # specify that here we are performing binary class training
    # binary_mode = 0 (normal vs NNV) / 1 (normal vs NV) / 2 (NNV vs NV)
    cfg_in.binary_class = True
    cfg_in.binary_mode = 1

    # whether or not to make the training set balanced - note this will give you imbalanced test set
    cfg_in.balanced = False

    # specify how many times to train the repeated models
    cfg_in.n_repeats = 10

    # now start the repeated training
    cfg_in.bool_debug = False
    cfg_in.str_arch = 'arch_025'
    run_repeated_training_binary(cfg_in)
