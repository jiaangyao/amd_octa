import pathlib
import copy

import numpy as np


def initialize_config_preproc(cfg_in):
    cfg = copy.deepcopy(cfg_in)

    # fix the paths...
    if cfg.user == 'jyao':
        cfg.d_data = pathlib.Path('/home/jyao/local/data/amd_octa/FinalData/')
        cfg.d_preproc = pathlib.Path('/home/jyao/local/data/amd_octa/PreprocData/')
        cfg.d_model = pathlib.Path('/home/jyao/local/data/amd_octa/trained_models/')
        cfg.d_csv = pathlib.Path('/home/jyao/local/data/amd_octa')

    elif cfg.user == 'kavi':
        cfg.d_data = pathlib.Path('/home/kavi/Downloads/Data/')
        cfg.d_preproc = pathlib.Path('/home/kavi/Downloads/PreprocData/')
        cfg.d_model = pathlib.Path('/home/kavi/Downloads/amd_octa_data/trained_models/')
        cfg.d_csv = pathlib.Path('/home/kavi/Downloads/amd_octa_data/')

    else:
        raise ValueError("Unknown user")

    if not cfg.d_preproc.exists():
        cfg.d_preproc.mkdir(exist_ok=True, parents=True)

    # cfg.f_csv = 'DiseaseLabelsThrough305.csv'
    cfg.f_csv = 'FeatureLabelsFullFinal.csv'

    cfg.vec_all_str_feature = ['disease', 'IRF/SRF', 'Scar', 'GA', 'CNV', 'Large PED']

    cfg.num_octa = 5
    cfg.str_angiography = 'Angiography'
    cfg.str_structure = 'Structure'
    cfg.str_bscan = 'B-Scan'

    cfg.vec_str_layer = ['Deep', 'Avascular', 'ORCC', 'Choriocapillaris', 'Choroid']
    cfg.vec_str_layer_bscan3d = ['1', '2', '3', '4', '5']
    cfg.dict_layer_order = {'Deep': 0,
                            'Avascular': 1,
                            'ORCC': 2,
                            'Choriocapillaris': 3,
                            'Choroid': 4}
    cfg.dict_layer_order_bscan3d = {'1': 0,
                                    '2': 1,
                                    '3': 2,
                                    '4': 3,
                                    '5': 4}
    cfg.str_bscan_layer = 'Flow'
    cfg.dict_str_patient_label = {}

    cfg.downscale_size = [256, 256]
    cfg.downscale_size_bscan = [450, 300]
    cfg.crop_size = [int(np.round(cfg.downscale_size_bscan[0] * 1.5 / 7.32)),
                     int(np.round(cfg.downscale_size_bscan[0] * 1.8 / 7.32))]

    # default flag for binary class
    if 'binary_class' not in cfg.keys():
        cfg.binary_class = False

    return cfg


def initialize_config_split(cfg_in):
    # make a copy of input feature
    cfg = copy.deepcopy(cfg_in)

    if cfg.str_feature == 'disease':
        cfg.vec_str_labels = ['Normal', 'NNV AMD', 'NV AMD']
    elif cfg.str_feature in ['IRF/SRF', 'Scar', 'GA', 'CNV', 'Large PED']:
        cfg.vec_str_labels = ['Not Present', 'Possible', 'Present']
    else:
        raise ValueError('Unknown feature type')

    # change the values
    cfg.str_healthy = 'Normal'
    cfg.label_healthy = 0
    cfg.str_dry_amd = 'Dry AMD'
    cfg.label_dry_amd = 1
    cfg.str_cnv = 'CNV'
    cfg.label_cnv = 2
    cfg.num_classes = 3

    # correct for the various labels if performing binary training
    if cfg.binary_class:
        cfg.num_classes = 2
        if cfg.binary_mode == 0:
            cfg.label_healthy = 0
            cfg.label_dry_amd = 1

        elif cfg.binary_mode == 1:
            cfg.label_healthy = 0
            cfg.label_cnv = 1

        elif cfg.binary_mode == 2:
            cfg.label_dry_amd = 0
            cfg.label_cnv = 1

        else:
            raise Exception('Undefined binary mode')

    # default flags for splitting the data
    cfg.per_train = 0.6
    cfg.per_valid = 0.2
    cfg.per_test = 0.2

    # default flags for cross validation
    if 'cv_mode' not in cfg.keys():
        cfg.cv_mode = False
    cfg.num_cv_fold = 5 if cfg.cv_mode else None

    # default flag for oversample method
    if 'oversample' not in cfg.keys():
        cfg.oversample = False
    cfg.oversample_method = 'smote' if cfg.oversample else None

    # default flag for the random seed
    if 'use_random_seed' not in cfg.keys():
        cfg.use_random_seed = True
    cfg.random_seed = 68 if cfg.use_random_seed else None

    return cfg


def initialize_config_training(cfg_in, bool_debug=False):
    cfg = copy.deepcopy(cfg_in)

    # now add the default training parameters to the configuration
    cfg.n_epoch = 1000
    cfg.batch_size = 8
    cfg.es_patience = 20
    cfg.es_min_delta = 1e-5
    cfg.lr = 5e-5
    cfg.lam = 1e-5
    cfg.overwrite = True

    if bool_debug:
        cfg.n_epoch = 1
        cfg.es_patience = 1
        cfg.overwrite = False

    return cfg
