import copy
import glob
import re

import numpy as np
from skimage import io, transform, color
import pathlib
from save_load.load_csv import load_csv_params


def data_loading(vec_idx_patient, cfg):
    if cfg.load_mode == 'folder':
        if not cfg.binary_class:
            print('\nLoading data from normal patients')
            x_healthy, y_healthy, vec_str_healthy_patient = load_specific_label_folder(vec_idx_patient, cfg.str_healthy,
                                                                                       cfg.label_healthy, cfg)
            print('Total number of healthy patients: {}\n'.format(x_healthy[0].shape[0]))

            print('\nLoading data from dry AMD patients')
            x_dry_amd, y_dry_amd, vec_str_dry_amd_patient = load_specific_label_folder(vec_idx_patient, cfg.str_dry_amd,
                                                                                       cfg.label_dry_amd, cfg)
            print('Total number of dry AMD patients: {}\n'.format(x_dry_amd[0].shape[0]))

            print('\nLoading data from CNV patients')
            x_cnv, y_cnv, vec_str_cnv_patient = load_specific_label_folder(vec_idx_patient, cfg.str_cnv, cfg.label_cnv,
                                                                           cfg)
            print('Total number of CNV patients: {}\n'.format(x_cnv[0].shape[0]))

            cfg.n_dry_amd = x_dry_amd[0].shape[0]
            cfg.n_cnv = x_cnv[0].shape[0]
            cfg.n_healthy = x_healthy[0].shape[0]

            # unpack once more
            x_angiography = np.concatenate((x_healthy[0], x_dry_amd[0], x_cnv[0]), axis=0)
            x_structure = np.concatenate((x_healthy[1], x_dry_amd[1], x_cnv[1]), axis=0)
            x_bscan = np.concatenate((x_healthy[2], x_dry_amd[2], x_cnv[2]), axis=0)
            x_bscan3d = np.concatenate((x_healthy[3], x_dry_amd[3], x_cnv[3]), axis=0)

            y = np.concatenate((y_healthy, y_dry_amd, y_cnv), axis=0)

            cfg.vec_str_patient = np.concatenate(
                (vec_str_healthy_patient, vec_str_dry_amd_patient, vec_str_cnv_patient), axis=0)

        else:
            # TODO: code below is ugly...
            if cfg.binary_mode == 0:
                print('\nLoading data from normal patients')
                x_healthy, y_healthy, vec_str_healthy_patient = load_specific_label_folder(vec_idx_patient,
                                                                                           cfg.str_healthy,
                                                                                           cfg.label_healthy, cfg)
                print('Total number of healthy patients: {}\n'.format(x_healthy[0].shape[0]))

                print('\nLoading data from dry AMD patients')
                x_dry_amd, y_dry_amd, vec_str_dry_amd_patient = load_specific_label_folder(vec_idx_patient,
                                                                                           cfg.str_dry_amd,
                                                                                           cfg.label_dry_amd, cfg)
                print('Total number of dry AMD patients: {}\n'.format(x_dry_amd[0].shape[0]))

                x_angiography = np.concatenate((x_healthy[0], x_dry_amd[0]), axis=0)
                x_structure = np.concatenate((x_healthy[1], x_dry_amd[1]), axis=0)
                x_bscan = np.concatenate((x_healthy[2], x_dry_amd[2]), axis=0)
                x_bscan3d = np.concatenate((x_healthy[3], x_dry_amd[3]), axis=0)
                y = np.concatenate((y_healthy, y_dry_amd), axis=0)

                cfg.vec_str_patient = np.concatenate((vec_str_healthy_patient, vec_str_dry_amd_patient), axis=0)

            elif cfg.binary_mode == 1:
                print('\nLoading data from normal patients')
                x_healthy, y_healthy, vec_str_healthy_patient = load_specific_label_folder(vec_idx_patient,
                                                                                           cfg.str_healthy,
                                                                                           cfg.label_healthy, cfg)
                print('Total number of healthy patients: {}\n'.format(x_healthy[0].shape[0]))

                print('\nLoading data from CNV patients')
                x_cnv, y_cnv, vec_str_cnv_patient = load_specific_label_folder(vec_idx_patient, cfg.str_cnv,
                                                                               cfg.label_cnv, cfg)
                print('Total number of CNV patients: {}\n'.format(x_cnv[0].shape[0]))

                x_angiography = np.concatenate((x_healthy[0], x_cnv[0]), axis=0)
                x_structure = np.concatenate((x_healthy[1], x_cnv[1]), axis=0)
                x_bscan = np.concatenate((x_healthy[2], x_cnv[2]), axis=0)
                x_bscan3d = np.concatenate((x_healthy[3], x_cnv[3]), axis=0)
                y = np.concatenate((y_healthy, y_cnv), axis=0)

                cfg.vec_str_patient = np.concatenate((vec_str_healthy_patient, vec_str_cnv_patient), axis=0)

            elif cfg.binary_mode == 2:
                print('\nLoading data from dry AMD patients')
                x_dry_amd, y_dry_amd, vec_str_dry_amd_patient = load_specific_label_folder(vec_idx_patient,
                                                                                           cfg.str_dry_amd,
                                                                                           cfg.label_dry_amd, cfg)
                print('Total number of dry AMD patients: {}\n'.format(x_dry_amd[0].shape[0]))

                print('\nLoading data from CNV patients')
                x_cnv, y_cnv, vec_str_cnv_patient = load_specific_label_folder(vec_idx_patient, cfg.str_cnv,
                                                                               cfg.label_cnv, cfg)
                print('Total number of CNV patients: {}\n'.format(x_cnv[0].shape[0]))

                x_angiography = np.concatenate((x_dry_amd[0], x_cnv[0]), axis=0)
                x_structure = np.concatenate((x_dry_amd[1], x_cnv[1]), axis=0)
                x_bscan = np.concatenate((x_dry_amd[2], x_cnv[2]), axis=0)
                x_bscan3d = np.concatenate((x_dry_amd[3], x_cnv[3]), axis=0)
                y = np.concatenate((y_dry_amd, y_cnv), axis=0)

                cfg.vec_str_patient = np.concatenate((vec_str_dry_amd_patient, vec_str_cnv_patient), axis=0)

            else:
                raise Exception('Undefined mode for binary classification')

        X = [x_angiography, x_structure, x_bscan, x_bscan3d]

    elif cfg.load_mode == 'csv':
        if cfg.d_csv is None or cfg.f_csv is None:
            raise Exception('Need to provide path of csv file if using csv load mode')

        vec_str_patient_id, _, _, _, _, _ = load_csv_params(cfg, bool_mode_full=False)

        X, vec_str_patients, vec_out_csv_str = load_all_data_csv(vec_idx_patient, vec_str_patient_id, cfg)

        cfg.vec_str_patients = vec_str_patients
        cfg.vec_out_csv_str = vec_out_csv_str

        y = None

    else:
        raise Exception('Undefined load mode')

    return X, y


def generate_labels(str_feature, cfg, bool_append_csv_to_cfg=False):
    cfg_local = copy.deepcopy(cfg)
    cfg_local.str_feature = str_feature

    if cfg_local.load_mode == 'csv':
        if cfg_local.d_csv is None or cfg_local.f_csv is None:
            raise Exception('Need to provide path of csv file if using csv load mode')
        if cfg_local.str_feature not in cfg_local.vec_all_str_feature:
            raise Exception('Invalid feature label provided')

        _, vec_OD_feature, vec_OS_feature, pd_csv, out_csv, vec_csv_col = \
            load_csv_params(cfg_local, bool_mode_full=True)

        # now generate the labels corresponding to the correct feature
        y, vec_out_csv_idx = _load_all_data_label_csv(cfg_local.vec_out_csv_str, vec_OD_feature, vec_OS_feature,
                                                      vec_csv_col)

        # perform sanity check using the input csv structure
        for i in range(len(vec_out_csv_idx)):
            idx_csv_out = vec_out_csv_idx[i]
            y_true_curr = pd_csv.iat[idx_csv_out[0], idx_csv_out[1]]

            if str_feature == 'disease':
                if y_true_curr == 0:
                    y_true_curr = np.nan
                else:
                    y_true_curr -= 1

            if not np.allclose(y[i], y_true_curr) and not np.all(np.isnan([y_true_curr, y[i]])):
                raise ValueError("These should be equal")

        if bool_append_csv_to_cfg:
            cfg.str_feature = str_feature
            cfg.vec_out_csv_idx = vec_out_csv_idx

            cfg.pd_csv = pd_csv.copy()
            cfg.out_csv = out_csv
            cfg.vec_csv_col = vec_csv_col

    else:
        raise NotImplementedError("Unsupported mode")

    return y


def correct_data_label(X, y, cfg, bool_use_ref_label=False, y_label_ref=False):
    """
    Perform correction to the full X and y data based on flags set in the configuration

    :param X:
    :param y:
    :param cfg:

    :return:
    """
    # make local copies of the input variables
    [x_angiography, x_structure, x_bscan, x_bscan3d] = X
    x_angiography_local = x_angiography.copy()
    x_structure_local = x_structure.copy()
    x_bscan_local = x_bscan.copy()
    x_bscan3d_local = x_bscan3d.copy()

    y_local = y.copy()

    # also make local copies of the hyperparameters
    vec_str_patients_local = copy.deepcopy(cfg.vec_str_patients)
    vec_out_csv_str_local = copy.deepcopy(cfg.vec_out_csv_str)
    vec_out_csv_idx_local = copy.deepcopy(cfg.vec_out_csv_idx)

    # check for NaN in the generated label
    if np.any(np.isnan(y)) or bool_use_ref_label:
        # obtain the labels that are actually valid
        if bool_use_ref_label:
            idx_valid_label = np.logical_not((np.isnan(y_label_ref)))
        else:
            idx_valid_label = np.logical_not((np.isnan(y_local)))

        # now correct for the data variables
        x_angiography_local = x_angiography_local[idx_valid_label, ...]
        x_structure_local = x_structure_local[idx_valid_label, ...]
        x_bscan_local = x_bscan_local[idx_valid_label, ...]
        x_bscan3d_local = x_bscan3d_local[idx_valid_label, ...]

        y_local = y_local[idx_valid_label, ...]

        # now correct for hyperparameter variables
        vec_str_patients_local_temp = []
        vec_out_csv_str_local_temp = []
        vec_out_csv_idx_local_temp = []
        for i in range(len(vec_str_patients_local)):
            if idx_valid_label[i]:
                vec_str_patients_local_temp.append(vec_str_patients_local[i])
                vec_out_csv_str_local_temp.append(vec_out_csv_str_local[i])
                vec_out_csv_idx_local_temp.append(vec_out_csv_idx_local[i])
        vec_str_patients_local = vec_str_patients_local_temp
        vec_out_csv_str_local = vec_out_csv_str_local_temp
        vec_out_csv_idx_local = vec_out_csv_idx_local_temp

        # sanity check
        if not x_angiography_local.shape[0] == x_structure_local.shape[0] == x_bscan_local.shape[0] == x_bscan3d_local.shape[0]:
            raise ValueError("These should be equal")
        if not x_angiography_local.shape[0] == y_local.shape[0]:
            raise ValueError("These should be equal")
        if not len(vec_str_patients_local) == len(vec_out_csv_str_local) == len(vec_out_csv_idx_local):
            raise ValueError("These should be equal")
        if bool_use_ref_label and not y_local.shape[0] == y_label_ref[idx_valid_label].shape[0]:
            raise ValueError("These should be equal ")

    # if mode is not binary class then check for label consistency and correct any identified inconsistencies
    if not cfg.binary_class:
        cfg.y_unique_label = np.unique(y_local)

        # if three unique labels and feature type is disease then do nothing
        if len(np.unique(y_local)) == 3 and cfg.str_feature == 'disease':
            pass

        # check if there are only two labels present, which is the case for many features
        elif len(np.unique(y_local)) == 2 and cfg.str_feature != 'disease':
            cfg.num_classes = 2
            cfg.binary_class = True

            vec_str_labels_temp = []
            for i in range(cfg.num_classes):
                vec_str_labels_temp.append(cfg.vec_str_labels[int(np.unique(y_local)[i])])
            cfg.vec_str_labels = vec_str_labels_temp

            # correct for labels where there are skips
            y_local_temp = y_local.copy()
            if not np.all(np.unique(y_local) == np.arange(0, cfg.num_classes)):
                for i in range(cfg.num_classes):
                    y_local_temp[y_local_temp == np.unique(y_local)[i]] = np.arange(0, cfg.num_classes)[i]
            y_local = y_local_temp

        elif len(np.unique(y_local)) == 2 and cfg.str_feature == 'disease':
            raise Exception('There should be three disease labels')

        elif len(np.unique(y_local)) == 4:
            raise Exception('Too many labels')

        else:
            raise ValueError("Unknown failure mode")

    # In the case of binary mode then also have to correct for the data for training
    else:
        cfg.y_unique_label = np.arange(0, 2, 1)
        y_local_temp = y_local.copy()
        if cfg.binary_mode == 0:
            idx_label_0 = y_local_temp == 0
            idx_label_1 = y_local_temp == 1
            cfg.vec_str_labels = ['Normal', 'NNV AMD']

        elif cfg.binary_mode == 1:
            idx_label_0 = y_local_temp == 0
            idx_label_1 = y_local_temp == 2
            cfg.vec_str_labels = ['Normal', 'NV AMD']

        elif cfg.binary_mode == 2:
            idx_label_0 = y_local_temp == 1
            idx_label_1 = y_local_temp == 2
            cfg.vec_str_labels = ['NNV AMD', 'NV AMD']

        else:
            raise ValueError("Unknown mode")

        X_angio_label0 = x_angiography_local[idx_label_0, ...]
        X_struct_label0 = x_structure_local[idx_label_0, ...]
        X_bscan_label0 = x_bscan_local[idx_label_0, ...]
        X_bscan3d_label0 = x_bscan3d_local[idx_label_0, ...]
        y_label0 = np.zeros_like(y_local[idx_label_0])

        X_angio_label1 = x_angiography_local[idx_label_1, ...]
        X_struct_label1 = x_structure_local[idx_label_1, ...]
        X_bscan_label1 = x_bscan_local[idx_label_1, ...]
        X_bscan3d_label1 = x_bscan3d_local[idx_label_1, ...]
        y_label1 = np.ones_like(y_local[idx_label_1])

        idx_full = np.arange(0, len(vec_str_patients_local), 1)
        idx_binary_label0 = idx_full[idx_label_0]
        idx_binary_label1 = idx_full[idx_label_1]
        idx_binary = np.concatenate([idx_binary_label0, idx_binary_label1])
        idx_binary_sort = np.argsort(idx_binary)
        idx_binary = idx_binary[idx_binary_sort]

        # piece together the data from the binary classes
        x_angiography_local = np.concatenate([X_angio_label0, X_angio_label1])[idx_binary_sort, ...]
        x_structure_local = np.concatenate([X_struct_label0, X_struct_label1])[idx_binary_sort, ...]
        x_bscan_local = np.concatenate([X_bscan_label0, X_bscan_label1])[idx_binary_sort, ...]
        x_bscan3d_local = np.concatenate([X_bscan3d_label0, X_bscan3d_label1])[idx_binary_sort, ...]

        y_local = np.concatenate([y_label0, y_label1])[idx_binary_sort]

        # now also fix the indexing in the metadata
        vec_str_patients_local_temp = []
        vec_out_csv_idx_local_temp = []
        vec_out_csv_str_local_temp = []
        for i in range(len(vec_str_patients_local)):
            if i in idx_binary:
                vec_str_patients_local_temp.append(vec_str_patients_local[i])
                vec_out_csv_idx_local_temp.append(vec_out_csv_idx_local[i])
                vec_out_csv_str_local_temp.append(vec_out_csv_str_local[i])
        vec_str_patients_local = vec_str_patients_local_temp
        vec_out_csv_str_local = vec_out_csv_str_local_temp
        vec_out_csv_idx_local = vec_out_csv_idx_local_temp

        # sanity check
        if not x_angiography_local.shape[0] == x_structure_local.shape[0] == x_bscan_local.shape[0] == x_bscan3d_local.shape[0]:
            raise ValueError("These should be equal")
        if not x_angiography_local.shape[0] == y_local.shape[0]:
            raise ValueError("These should be equal")
        if not len(vec_str_patients_local) == len(vec_out_csv_str_local) == len(vec_out_csv_idx_local):
            raise ValueError("These should be equal")

    X_out = [x_angiography_local, x_structure_local, x_bscan_local, x_bscan3d_local]
    y_out = y_local

    cfg.vec_str_patients = vec_str_patients_local
    cfg.vec_out_csv_idx = vec_out_csv_idx_local
    cfg.vec_out_csv_str = vec_out_csv_str_local

    # sanity check
    if np.any(np.isnan(y_out)):
        raise ValueError("Should not be NaN")

    return X_out, y_out


def load_all_data_csv(vec_idx, vec_str_patient_id, cfg):
    """
    Functional wrapper for loading data from all patients using function below

    :param vec_idx: list in the form of [start_idx, end_idx]
    :param cfg: configuration file set by the user
    :return:
    """

    x, vec_str_patients, vec_out_csv_str = \
        _load_all_data_csv(vec_idx, vec_str_patient_id, cfg.d_data, cfg.downscale_size, cfg.downscale_size_bscan,
                           cfg.crop_size, cfg.num_octa, cfg.str_angiography, cfg.str_structure,
                           cfg.str_bscan, cfg.vec_str_layer, cfg.vec_str_layer_bscan3d,
                           cfg.str_bscan_layer, cfg.dict_layer_order, cfg.dict_layer_order_bscan3d)

    return x, vec_str_patients, vec_out_csv_str


def _load_all_data_csv(vec_idx, vec_str_patient_id, d_data, downscale_size, downscale_size_bscan,
                       crop_size, num_octa, str_angiography, str_structure,
                       str_bscan, vec_str_layer, vec_str_layer_bscan3d,
                       str_bscan_layer, dict_layer_order, dict_layer_order_bscan3d):

    """
    Load all data from all patients without assigning the class label yet

    :param vec_idx: list in the form of [start_idx, end_idx]
    :param vec_str_patient_id:
    :param pathlib.Path d_data: directory to the data
    :param list downscale_size: desired shape after downscaling the images, e.g. [350, 350]
    :param list downscale_size_bscan: desired shape after downscaling the bscan images, e.g. [350, 350]
    :param list crop_size: desired number of pixels to exclude from analysis for bscan images, e.g. [50, 60]
    :param num_octa: number of OCTA images per patient, e.g. 5
    :param str_angiography: identifier for structural angiography images in the filename
    :param str_structure: identifier for structural OCT images in the filename
    :param str_bscan: identifier for structural b-scan images in the filename
    :param vec_str_layer: list of strings that contain the relevant layers to be used for training
    :param vec_str_layer_bscan3d: list of strings that contain the relevant bscan images to be used for training
    :param str_bscan_layer: string that contains the type of b-scan images to be used in filename, e.g. Flow
    :param dict_layer_order: dictionary that contains the order in which the different layers will be organized
    :param dict_layer_order_bscan3d: dictionary that contains the order in which the bscans cubes will be organized

    :return: a tuple in the form [x_angiography, x_structure, x_bscan], vec_str_patient, where each of x_class
    contains images from a single type of image and vec_str_patient would correspond to absolute
    """
    # create the empty lists for holding the variables
    x_angiography = []
    x_structure = []
    x_bscan = []
    x_bscan3d = []

    # create a list to append all patients
    vec_str_patient = []
    vec_out_csv_str = []

    # create a list of all possible indices
    vec_full_idx = np.arange(vec_idx[0], vec_idx[1] + 1, 1)

    # Loop through all the patients
    for i in range(len(vec_full_idx)):
        vec_f_image = glob.glob(str(d_data / '{}'.format(vec_full_idx[i]) / '*' / 'OCTA' / '*.bmp'), recursive=True)
        vec_f_imageBscan3d = glob.glob(str(d_data / '{}'.format(vec_full_idx[i]) / '*' / '*.tiff'), recursive=True)

        # try old pattern again also
        if not vec_f_image:
            vec_f_image = glob.glob(str(d_data / '{}'.format(vec_full_idx[i]) / '*' / '2D Layers' / '*.bmp'),
                                    recursive=True)
        if vec_f_image:
            print("Loading data from patient {}".format(vec_full_idx[i]))
        else:
            print("Data not available for patient {}, skipping...".format(vec_full_idx[i]))
            continue

        if vec_f_imageBscan3d:
            print("Loading 3d bscan data from patient {}".format(vec_full_idx[i]))
        
        else:
            print("Data (bscan3d) not available for patient {}, skipping...".format(vec_full_idx[i]))
            continue

        packed_x_curr, str_eye = _package_data(vec_f_image, vec_f_imageBscan3d, downscale_size, downscale_size_bscan,
                                               crop_size, num_octa,
                                               str_angiography, str_structure, str_bscan,
                                               vec_str_layer, vec_str_layer_bscan3d, str_bscan_layer,
                                               dict_layer_order, dict_layer_order_bscan3d)

        if packed_x_curr is None:
            print("Unable to process data for patient {}, skipping...".format(vec_full_idx[i]))
            continue

        # test for the label independently as well
        rel_idx_patient_id = np.where(vec_str_patient_id == vec_full_idx[i])[0][0]

        # now unpack the data
        if len(packed_x_curr) == 2:
            for j in range(len(packed_x_curr)):
                if str_eye[j] not in ['OD', 'OS']:
                    raise Exception('Invalid eye label encountered')

                x_angiography.append(packed_x_curr[j][0])
                x_structure.append(packed_x_curr[j][1])
                x_bscan.append(packed_x_curr[j][2])
                x_bscan3d.append(packed_x_curr[j][3])

                # append to list of patients
                str_patient = "Patient {}/{}".format(vec_full_idx[i], str_eye[j])
                vec_str_patient.append(str_patient)

                vec_out_csv_str.append([rel_idx_patient_id, str_eye[j]])

        else:
            if str_eye not in ['OD', 'OS']:
                raise Exception('Invalid eye label encountered')

            x_angiography.append(packed_x_curr[0])
            x_structure.append(packed_x_curr[1])
            x_bscan.append(packed_x_curr[2])
            x_bscan3d.append(packed_x_curr[3])

            # append to list of patients
            str_patient = "Patient {}/{}".format(vec_full_idx[i], str_eye)
            vec_str_patient.append(str_patient)

            vec_out_csv_str.append([rel_idx_patient_id, str_eye])

    x_angiography = np.stack(x_angiography, axis=0)
    x_structure = np.stack(x_structure, axis=0)
    x_bscan = np.stack(x_bscan, axis=0)
    x_bscan3d = np.stack(x_bscan3d, axis=0)

    # sanity check
    if not x_angiography.shape[0] == x_structure.shape[0] == x_bscan.shape[0] == x_bscan3d.shape[0]:
        raise ValueError("These should be equal")
    if not x_angiography.shape[0] == len(vec_str_patient) == len(vec_out_csv_str):
        raise ValueError("These should be equal")

    return [x_angiography, x_structure, x_bscan, x_bscan3d], vec_str_patient, vec_out_csv_str


def _load_all_data_label_csv(vec_out_csv_str, vec_OD_feature, vec_OS_feature, vec_csv_col):
    # create an empty list to hold all labels
    y = []

    # create a list of all valid entries in the csv file
    vec_out_csv_idx = []
    idx_col_OD_feature = vec_csv_col[-2]
    idx_col_OS_feature = vec_csv_col[-1]

    # Loop through all the patients
    for i in range(len(vec_out_csv_str)):
        csv_str_curr = vec_out_csv_str[i]

        if csv_str_curr[1] == 'OD':
            y_curr = vec_OD_feature[csv_str_curr[0]]
            vec_out_csv_idx.append([csv_str_curr[0], idx_col_OD_feature])
        elif csv_str_curr[1] == 'OS':
            y_curr = vec_OS_feature[csv_str_curr[0]]
            vec_out_csv_idx.append([csv_str_curr[0], idx_col_OS_feature])
        else:
            raise ValueError("Unknown mode provided")

        y.append(y_curr)

    y = np.stack(y, axis=0)

    return y, vec_out_csv_idx


def load_specific_label_folder(vec_idx_class, str_class, label_class, cfg):
    """
    Functional wrapper for loading a specific label using function below

    :param vec_idx_class: list in the form of [start_idx, end_idx]
    :param str_class: name of this class, e.g. normalPatient
    :param label_class: label assigned to this class, e.g. 0
    :param cfg: configuration file set by the user
    :return:
    """
    x_class, y_class, vec_str_class = _load_data_folder(vec_idx_class, str_class, label_class,
                                                        cfg.d_data, cfg.downscale_size, cfg.downscale_size_bscan,
                                                        cfg.crop_size, cfg.num_octa,
                                                        cfg.str_angiography, cfg.str_structure, cfg.str_bscan,
                                                        cfg.vec_str_layer, cfg.vec_str_layer_bscan3d, cfg.str_bscan_layer,
                                                        cfg.dict_layer_order, cfg.dict_layer_order_bscan3d)

    return x_class, y_class, vec_str_class


def _load_data_folder(vec_idx, str_class, label_class, d_data, downscale_size, downscale_size_bscan, crop_size, num_octa,
                      str_angiography, str_structure, str_bscan, vec_str_layer, vec_str_layer_bscan3d,
                      str_bscan_layer, dict_layer_order, dict_layer_order_bscan3d):
    """
    Load data of a specific class based on folder structure

    :param vec_idx: list in the form of [start_idx, end_idx]
    :param str_class: name of this class, e.g. normalPatient
    :param label_class: label assigned to this class, e.g. 0
    :param pathlib.Path d_data: directory to the data
    :param list downscale_size: desired shape after downscaling the images, e.g. [350, 350]
    :param list downscale_size_bscan: desired shape after downscaling the bscan images, e.g. [350, 350]
    :param list crop_size: desired number of pixels to exclude from analysis for bscan images, e.g. [50, 60]
    :param num_octa: number of OCTA images per patient, e.g. 5
    :param str_angiography: identifier for structural angiography images in the filename
    :param str_structure: identifier for structural OCT images in the filename
    :param str_bscan: identifier for structural b-scan images in the filename
    :param vec_str_layer: list of strings that contain the relevant layers to be used for training
    :param vec_str_layer_bscan3d: list of strings that contain the relevant bscan images to be used for training
    :param str_bscan_layer: string that contains the type of b-scan images to be used in filename, e.g. Flow
    :param dict_layer_order: dictionary that contains the order in which the different layers will be organized
    :param dict_layer_order_bscan3d: dictionary that contains the order in which the bscans cubes will be organized

    :return: a tuple in the form [x_angiography, x_structure, x_bscan], y, vec_str_patient, where each of x_class
    contains images from a single type of image, y would correspond to label of all patients and vec_str_patient
    would correspond to absolute
    """

    # create the empty lists for holding the variables
    x_angiography = []
    x_structure = []
    x_bscan = []
    x_bscan3d = []

    y = []

    # create a list to append all patients
    vec_str_patient = []

    # create a list of all possible indices
    vec_full_idx = np.arange(vec_idx[0], vec_idx[1] + 1, 1)

    # Loop through all the runs
    for i in range(len(vec_full_idx)):
        vec_f_image = glob.glob(str(d_data / str_class / '{}'.format(vec_full_idx[i]) / '*' / 'OCTA' / '*.bmp'),
                                recursive=True)
        vec_f_imageBscan3d = glob.glob(str(d_data / str_class / '{}'.format(vec_full_idx[i]) / '*' / '*.tiff'),
                                       recursive=True)

        # try old pattern again also
        if not vec_f_image:
            vec_f_image = glob.glob(str(d_data / '{}'.format(vec_full_idx[i]) / '*' / '2D Layers' / '*.bmp'),
                                    recursive=True)

        if vec_f_image:
            print("Loading data from patient {}".format(vec_full_idx[i]))
        else:
            print("Data not available for patient {}, skipping...".format(vec_full_idx[i]))
            continue

        if vec_f_imageBscan3d:
            print("Loading 3d bscan data from patient {}".format(vec_full_idx[i]))
        else:
            print("Data (bscan3d) not available for patient {}, skipping...".format(vec_full_idx[i]))
            continue

        packed_x_curr, str_eye = _package_data(vec_f_image, vec_f_imageBscan3d, downscale_size, downscale_size_bscan,
                                               crop_size, num_octa,
                                               str_angiography, str_structure, str_bscan,
                                               vec_str_layer, vec_str_layer_bscan3d, str_bscan_layer,
                                               dict_layer_order, dict_layer_order_bscan3d)

        if packed_x_curr is None:
            raise Exception("Unable to process data for patient {}, skipping...".format(vec_full_idx[i]))

        # now unpack the data
        if len(packed_x_curr) == 2:
            for j in range(len(packed_x_curr)):
                x_angiography.append(packed_x_curr[j][0])
                x_structure.append(packed_x_curr[j][1])
                x_bscan.append(packed_x_curr[j][2])
                x_bscan3d.append(packed_x_curr[j][3])

                # append the class label also
                y.append(label_class)

                # append to list of patients
                str_patient = "{}/Patient {}/{}".format(str_class, vec_full_idx[i], str_eye[j])
                vec_str_patient.append(str_patient)

        else:
            x_angiography.append(packed_x_curr[0])
            x_structure.append(packed_x_curr[1])
            x_bscan.append(packed_x_curr[2])
            x_bscan3d.append(packed_x_curr[3])

            # append the class label also
            y.append(label_class)

            # append to list of patients
            str_patient = "{}/Patient {}/{}".format(str_class, vec_full_idx[i], str_eye)
            vec_str_patient.append(str_patient)

    x_angiography = np.stack(x_angiography, axis=0)
    x_structure = np.stack(x_structure, axis=0)
    x_bscan = np.stack(x_bscan, axis=0)
    x_bscan3d = np.stack(x_bscan3d, axis=0)

    return [x_angiography, x_structure, x_bscan, x_bscan3d], y, vec_str_patient


def _package_data(vec_f_image, vec_f_imageBscan3d, downscale_size, downscale_size_bscan, crop_size, num_octa,
                  str_angiography, str_structure, str_bscan, vec_str_layer, vec_str_layer_bscan3d, str_bscan_layer,
                  dict_layer_order, dict_layer_order_bscan3d):
    """
    Organizes the angiography, OCT and b-scan images into a list of cubes for a single subject and also returns which
    eye it is. Difference from function below: contains logic that deal with cases where there are two eyes

    :param vec_f_image: list of absolute paths to individual images from a single subject
    :param vec_f_imageBscan3d: list of absolute paths to individual bscan images from a single subject
    :param downscale_size: desired final size of the loaded images, e.g. (256, 256)
    :param downscale_size_bscan: desired final size of the loaded bscan images, e.g. (256, 256)
    :param crop_size: desired number of pixels to exclude from analysis for bscan images, e.g. [50, 60]
    :param num_octa: number of OCTA images per patient, e.g. 5
    :param str_angiography: identifier for angiography images in the filename
    :param str_structure: identifier for structural OCT images in the filename
    :param str_bscan: identifier for b-scan OCT images in the filename
    :param vec_str_layer: list of strings that contain the relevant layers to be used for training
    :param vec_str_layer_bscan3d: list of strings that contain the relevant bscan images to be used for training
    :param str_bscan_layer: string that contains the type of b-scan images to be used in filename, e.g. Flow
    :param dict_layer_order: dictionary that contains the order in which the different layers will be organized
    :param dict_layer_order_bscan3d: dictionary that contains the order in which the bscans cubes will be organized

    :return: return a list in the form [packed_images, str_eye]. If both eyes are available, then each variable would
    be a list of cubes and strings; if only one eye is available, packed_images would be a cube and str_eye would be
    OD/OS; if nothing is available then both are None
    """

    # Test if the dataset has data from both eyes
    if any("/OD/" in s for s in vec_f_image) & any("/OS/" in s for s in vec_f_image):

        vec_f_image_OD = [s for s in vec_f_image if "/OD/" in s]
        vec_f_image_OS = [s for s in vec_f_image if "/OS/" in s]

        vec_f_imageBscan3d_OD = [s for s in vec_f_imageBscan3d if "/OD/" in s]
        vec_f_imageBscan3d_OS = [s for s in vec_f_imageBscan3d if "/OS/" in s]

        x_curr_OD = _form_cubes(vec_f_image_OD, vec_f_imageBscan3d_OD, num_octa, downscale_size, downscale_size_bscan, crop_size,
                                str_angiography, str_structure, str_bscan, vec_str_layer, vec_str_layer_bscan3d,
                                str_bscan_layer, dict_layer_order, dict_layer_order_bscan3d)

        x_curr_OS = _form_cubes(vec_f_image_OS, vec_f_imageBscan3d_OS, num_octa, downscale_size, downscale_size_bscan, crop_size,
                                str_angiography, str_structure, str_bscan, vec_str_layer, vec_str_layer_bscan3d,
                                str_bscan_layer, dict_layer_order, dict_layer_order_bscan3d)

        # Figure out if any of the single eye data is none
        if x_curr_OD is not None and x_curr_OS is not None:
            packed_x_curr = [x_curr_OD, x_curr_OS]
            str_eye = ['OD', 'OS']

        elif x_curr_OD is None and x_curr_OS is not None:
            packed_x_curr = x_curr_OS
            str_eye = 'OS'

        elif x_curr_OD is not None and x_curr_OS is None:
            packed_x_curr = x_curr_OD
            str_eye = 'OD'

        else:
            packed_x_curr = None
            str_eye = None

    else:
        x_curr = _form_cubes(vec_f_image, vec_f_imageBscan3d, num_octa, downscale_size, downscale_size_bscan, crop_size,
                             str_angiography, str_structure, str_bscan, vec_str_layer, vec_str_layer_bscan3d,
                             str_bscan_layer, dict_layer_order, dict_layer_order_bscan3d)

        packed_x_curr = x_curr

        if any("_OD_" in s for s in vec_f_image):
            str_eye = 'OD'

        elif any("_OS_" in s for s in vec_f_image):
            str_eye = 'OS'

        else:
            str_eye = None

    return packed_x_curr, str_eye


def _form_cubes(vec_f_image, vec_f_imageBscan3d, num_octa, downscale_size, downscale_size_bscan, crop_size,
                str_angiography, str_structure, str_bscan, vec_str_layer, vec_str_layer_bscan3d, str_bscan_layer,
                dict_layer_order, dict_layer_order_bscan3d):
    """
    Organizes the angiography, OCT and b-scan images into a list of cubes for a single subject

    :param vec_f_image: list of absolute paths to individual images from a single subject
    :param vec_f_imageBscan3d: list of absolute paths to individual bscan images from a single subject
    :param num_octa: number of OCTA images per patient, e.g. 5
    :param downscale_size: desired final size of the loaded images, e.g. (256, 256)
    :param downscale_size_bscan: desired final size of the loaded bscan images, e.g. (256, 256)
    :param crop_size: desired number of pixels to exclude from analysis for bscan images, e.g. [50, 60]
    :param str_angiography: identifier for angiography images in the filename
    :param str_structure: identifier for structural OCT images in the filename
    :param str_bscan: identifier for b-scan OCT images in the filename
    :param vec_str_layer: list of strings that contain the relevant layers to be used for training
    :param vec_str_layer_bscan3d: list of strings that contain the relevant bscan images to be used for training
    :param str_bscan_layer: string that contains the type of b-scan images to be used in filename, e.g. Flow
    :param dict_layer_order: dictionary that contains the order in which the different layers will be organized
    :param dict_layer_order_bscan3d: dictionary that contains the order in which the bscans cubes will be organized

    :return: a list that contains loaded angiography, OCT, and b-scan images
    """
    # TODO: this function is heavily hardcoded at this point...

    vec_vol_f_image_angiography_curr = {}
    vec_vol_f_image_structure_curr = {}
    vec_vol_f_image_bscan_curr = {}
    vec_vol_f_image_bscan3d_curr = {}

    count_angiography_hit = 0
    count_structure_hit = 0
    count_bscan_hit = 0
    for f_image in vec_f_image:
        p_f_image = pathlib.Path(f_image)
        p_f_image_filename = p_f_image.name

        for str_image_type in [str_angiography, str_structure, str_bscan]:
            if str_image_type == str_bscan:
                re_pattern_bscan = '.*{} {}.bmp'.format(str_image_type, str_bscan_layer)

                re_hits = re.findall(re_pattern_bscan, p_f_image_filename, re.I)

                if re_hits:
                    vec_vol_f_image_bscan_curr[len(vec_vol_f_image_bscan_curr)] = f_image
                    count_bscan_hit += 1
            else:
                for str_layer in vec_str_layer:
                    re_pattern_curr = '.*{}_{}.bmp'.format(str_image_type, str_layer)

                    re_hits = re.findall(re_pattern_curr, p_f_image_filename, re.I)

                    if re_hits:
                        if str_image_type == str_angiography:
                            vec_vol_f_image_angiography_curr[dict_layer_order[str_layer]] = f_image
                            count_angiography_hit += 1
                        else:
                            vec_vol_f_image_structure_curr[dict_layer_order[str_layer]] = f_image
                            count_structure_hit += 1

    if (count_angiography_hit != len(vec_str_layer)) or (count_structure_hit != len(vec_str_layer)) or (count_bscan_hit != 1):
        raise Exception('Failed to locate all the OCTA images')

    count_bscan3d_hit = 0
    for f_image_bscan3d in vec_f_imageBscan3d:
        p_f_image_bscan3d = pathlib.Path(f_image_bscan3d)
        p_f_image_bscan3d_filename = p_f_image_bscan3d.name

        #for 3d bscan cube
        for str_layer_bscan3d in vec_str_layer_bscan3d:
            if  p_f_image_bscan3d_filename == '{}.tiff'.format(str_layer_bscan3d):
                vec_vol_f_image_bscan3d_curr[dict_layer_order_bscan3d[str_layer_bscan3d]] = f_image_bscan3d
                count_bscan3d_hit += 1

    # in cases where no images were identified, try again with different pattern
    if count_bscan3d_hit == 0:
        # obtain the date strings at the end of the file
        vec_img_timestamp = []
        re_pattern_timestamp = '_O[DS]_(.*).tiff'
        for f_image_bscan3d in vec_f_imageBscan3d:
            p_f_image_bscan3d = pathlib.Path(f_image_bscan3d)
            p_f_image_bscan3d_filename = p_f_image_bscan3d.name

            str_img_timestamp = re.search(re_pattern_timestamp, p_f_image_bscan3d_filename).group(1)
            vec_img_timestamp.append(int(str_img_timestamp))
        vec_img_timestamp = np.stack(vec_img_timestamp, axis=-1)
        vec_img_timestamp = np.sort(vec_img_timestamp)

        # do actual detection and addition to the dictionary
        for f_image_bscan3d in vec_f_imageBscan3d:
            p_f_image_bscan3d = pathlib.Path(f_image_bscan3d)
            p_f_image_bscan3d_filename = p_f_image_bscan3d.name

            for idx_img_timestamp in range(len(vec_img_timestamp)):
                img_timestamp = vec_img_timestamp[idx_img_timestamp]
                re_pattern_curr = '.*_{}.tiff'.format(img_timestamp)
                re_hits = re.findall(re_pattern_curr, p_f_image_bscan3d_filename, re.I)
                if re_hits:
                    vec_vol_f_image_bscan3d_curr[idx_img_timestamp] = f_image_bscan3d
                    count_bscan3d_hit += 1

    if count_bscan3d_hit != len(vec_str_layer_bscan3d):
        raise Exception('Failed to locate all the 3D bscan images')

    # Test if we have all data from this subject
    # TODO: think about using masking if cases where we don't have enough data
    # why is masking needed: some patients don't have as many images as others
    # why is masking challenging: don't think Keras supports conv3D with masking just yet
    #   so need to write custom layer that supports masking;
    #   see https://israelg99.github.io/2017-02-27-Grayscale-PixelCNN-with-Keras/
    # alternative method: apply the time distributed layer and treat each slice separately...
    #   but that might not be what we want here...

    if any(len(s) < num_octa for s in [vec_vol_f_image_angiography_curr, vec_vol_f_image_structure_curr]) \
            or not vec_vol_f_image_bscan_curr:
        print('Missing figures, skipping...')
        return None

    # once we are down with finding the appropriate path, try to load the images
    # TODO: number of channels is hard-coded now, fix that in the future
    vol_angiography_curr = np.zeros([downscale_size[0], downscale_size[1], num_octa, 1])
    vol_structure_curr = np.zeros([downscale_size[0], downscale_size[1], num_octa, 1])
    vol_bscan_curr = np.zeros([downscale_size_bscan[0], downscale_size_bscan[1], 1])


    _create_np_cubes(vol_angiography_curr, vec_vol_f_image_angiography_curr, downscale_size)
    _create_np_cubes(vol_structure_curr, vec_vol_f_image_structure_curr, downscale_size)
    _create_np_cubes(vol_bscan_curr, vec_vol_f_image_bscan_curr, downscale_size_bscan)

    if crop_size is None:
        vol_bscan3d_curr = np.zeros([downscale_size_bscan[0], downscale_size_bscan[1], num_octa, 1])
        _create_np_cubes(vol_bscan3d_curr, vec_vol_f_image_bscan3d_curr, downscale_size_bscan)

    else:
        vol_bscan3d_curr = np.zeros([downscale_size_bscan[0] - crop_size[0] - crop_size[1], downscale_size_bscan[1], num_octa, 1])
        _create_np_cubes(vol_bscan3d_curr, vec_vol_f_image_bscan3d_curr, downscale_size_bscan,
                         bool_crop=True, crop_size=crop_size)

    x_curr = [vol_angiography_curr, vol_structure_curr, vol_bscan_curr, vol_bscan3d_curr]

    return x_curr


def _create_np_cubes(np_cube, vec_vol_f_image, downscale_size, bool_crop=False, crop_size=None):
    """
    Packs loaded single-type (e.g. OCT) individual images into numpy tensors of shape (width, height, n_octa, 1).
    This would correspond to data from a single patient

    :param np_cube: numpy tensor of shape (width, height, n_octa, 1) that will be filled up with individual images
    :param vec_vol_f_image: a dictionary of the absolute paths to the individual images
    :param downscale_size: desired final size of the loaded images, e.g. (256, 256)
    """
    if len(np_cube.shape) == 4:
        for idx_layer, p_f_image_curr_layer in vec_vol_f_image.items():
            curr_img = _load_individual_image(str(p_f_image_curr_layer), downscale_size, bool_crop, crop_size)
            np_cube[:, :, idx_layer, :] = curr_img

    # TODO: code for B scan is ugly
    else:
        np_cube[:, :] = _load_individual_image(str(vec_vol_f_image[0]), downscale_size, bool_crop, crop_size)


def _load_individual_image(f_image, downscale_size, bool_crop=False, crop_size=None):
    """
    Loads an individual image into numpy array and perform resizing

    :param f_image: absolute path to a single image
    :param downscale_size: desired final size of the loaded images, e.g. (256, 256)
    :return: individual grayscale image of shape (width, height, 1)
    """
    img = io.imread(f_image, plugin='matplotlib').astype(np.float32)

    # Take care of images that are not grayscale
    if len(img.shape) >= 3:
        if len(img.shape) >= 4:
            raise Exception('We have a 4D vector as raw input')

        # If we have a RGB image
        if img.shape[-1] == 3:
            img = color.rgb2gray(img)

        # If we have RGBa image, disgard the last channel and only use
        if img.shape[-1] == 4:
            img = color.rgb2gray(img[:, :, 0:3])

    if np.max(img) > 1:
        img = img / 255

    # Note here you can't use numpy resize since the end result obtained is different from original image
    imgResized = transform.resize(img, (downscale_size[0], downscale_size[1], 1), anti_aliasing=True)

    # if we want to crop the image
    if bool_crop:
        imgResized_orig = imgResized.copy()
        crop_end = downscale_size[0] - crop_size[1]
        imgResized = imgResized[crop_size[0]:crop_end, :, :]

    return imgResized
