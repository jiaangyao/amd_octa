import glob
import re

import numpy as np
from skimage import io, transform, color
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from utils.context_management import temp_seed
from load_csv import load_csv_params


def preprocess(vec_idx_patient, cfg):
    # Exception detection
    if not cfg.binary_class:
        if not cfg.num_classes == 3:
            raise Exception('Three class classification specified but two classes requested')
    else:
        if not cfg.num_classes == 2:
            raise Exception('Binary classification specified but three classes requested')

    # load all data
    [x_angiography, x_structure, x_bscan, x_bscan3d], y = data_loading(vec_idx_patient, cfg)

    # split the data into training, validation and test set
    if not cfg.balanced:
        if cfg.use_random_seed:
            with temp_seed(cfg.random_seed):
                Xs, ys = _split_data_unbalanced(x_angiography, x_structure, x_bscan, x_bscan3d, y, cfg)
        else:
            Xs, ys = _split_data_unbalanced(x_angiography, x_structure, x_bscan, x_bscan3d, y, cfg)

        if cfg.oversample:
            if cfg.oversample_method == 'smote':
                x_train = Xs[0]
                x_angiography_train = x_train[0]
                x_structure_train = x_train[1]
                x_bscan_train = x_train[2]
                x_bscan3d_train = x_train[3]

                y_train = np.argmax(ys[0], axis=1)

                x_angiography_train_rs = x_angiography_train.reshape(x_angiography_train.shape[0], -1)
                x_structure_train_rs = x_structure_train.reshape(x_structure_train.shape[0], -1)
                x_bscan_train_rs = x_bscan_train.reshape(x_bscan_train.shape[0], -1)
                x_bscan3d_train_rs = x_bscan3d_train.reshape(x_bscan3d_train.shape[0], -1)

                sm = SMOTE()
                x_angiography_train_rs, y_train_rs = sm.fit_resample(x_angiography_train_rs, y_train)
                x_structure_train_rs, y_train_rs_alt = sm.fit_resample(x_structure_train_rs, y_train)
                x_bscan_train_rs, y_train_rs_alt_alt = sm.fit_resample(x_bscan_train_rs, y_train)
                x_bscan3d_train_rs, y_train_rs_alt_alt = sm.fit_resample(x_bscan3d_train_rs, y_train)

                angio_shape = [x_angiography_train_rs.shape[0]]
                angio_shape.extend(list(x_angiography_train.shape[1:]))

                structure_shape = [x_structure_train_rs.shape[0]]
                structure_shape.extend(list(x_structure_train.shape[1:]))

                bscan_shape = [x_bscan_train_rs.shape[0]]
                bscan_shape.extend(list(x_bscan_train.shape[1:]))

                bscan3d_shape = [x_bscan3d_train_rs.shape[0]]
                bscan3d_shape.extend(list(x_bscan3d_train.shape[1:]))

                x_angiography = x_angiography_train_rs.reshape(angio_shape)
                x_structure = x_structure_train_rs.reshape(structure_shape)
                x_bscan = x_bscan_train_rs.reshape(bscan_shape)
                x_bscan3d = x_bscan3d_train_rs.reshape(bscan3d_shape)

                if not (np.allclose(y_train_rs, y_train_rs_alt) and np.allclose(y_train_rs, y_train_rs_alt_alt)):
                    raise Exception("Issues with SMOTE")

                x_train = [x_angiography, x_structure, x_bscan]
                y_train = to_categorical(y_train_rs, num_classes=cfg.num_classes)

                Xs = [x_train, Xs[1], Xs[2]]
                ys = [y_train, ys[1], ys[2]]

                raise Exception("Not fully implemented")

            elif cfg.oversample_method == 'random':
                raise NotImplementedError

    else:
        if cfg.use_random_seed:
            with temp_seed(cfg.random_seed):
                Xs, ys = _split_data(x_angiography, x_structure, x_bscan, x_bscan3d, y, cfg)
        else:
            Xs, ys = _split_data(x_angiography, x_structure, x_bscan, x_bscan3d, y, cfg)

    return Xs, ys


def preprocess_cv(vec_idx_patient, cfg):
    """
    Cross validation mode of the preprocessing function

    :param vec_idx_patient: list containing start and end indices of the patients
    :param cfg: object holding all the training parameters

    :return:
    """
    # Exception detection
    if not cfg.binary_class:
        if not cfg.num_classes == 3:
            raise Exception('Three class classification specified but two classes requested')
    else:
        if not cfg.num_classes == 2:
            raise Exception('Binary classification specified but three classes requested')
    if not cfg.cv_mode:
        raise Exception("CV mode should be enabled")
    if cfg.num_cv_fold is None:
        raise Exception("Should specify the number of cv folds")
    if cfg.num_cv_fold != int(np.ceil(1 / cfg.per_test)):
        raise Exception("Number of cv folds should match the test set percentage")
    if (cfg.per_train + cfg.per_valid + cfg.per_test) != 1.0:
        raise Exception("Percentage of train, validation and test sets should add to 1")

    # load all data
    [x_angiography, x_structure, x_bscan, x_bscan3d], y = data_loading(vec_idx_patient, cfg)

    # split the data into training, validation and test set
    if not cfg.balanced:
        if cfg.use_random_seed:
            with temp_seed(cfg.random_seed):
                vec_Xs, vec_ys = _split_data_unbalanced_cv(x_angiography, x_structure, x_bscan, x_bscan3d, y, cfg)
        else:
            vec_Xs, vec_ys = _split_data_unbalanced_cv(x_angiography, x_structure, x_bscan, x_bscan3d, y, cfg)

        # Don't need to do oversample anymore in the future
        if cfg.oversample:
            raise NotImplementedError

    else:
        # In the dataset there are huge imbalances for the feature labels... I don't think doing balanced training would
        # be a good idea to be honest
        raise NotImplementedError

    return vec_Xs, vec_ys


def _split_data_unbalanced(x_angiography, x_structure, x_bscan, x_bscan3d, y, cfg):
    n_iter = 0
    while True:
        idx_permutation = np.random.permutation(x_angiography.shape[0])
        x_angiography_curr = x_angiography[idx_permutation, :, :, :, :]
        x_structure_curr = x_structure[idx_permutation, :, :, :, :]
        x_bscan_curr = x_bscan[idx_permutation, :, :, :]
        x_bscan3d_curr = x_bscan3d[idx_permutation, :, :, :, :]

        y_curr = y[idx_permutation]

        # split into train, validation and test
        n_train = int(np.ceil(len(idx_permutation) * cfg.per_train))
        n_valid = int(np.floor(len(idx_permutation) * cfg.per_valid))

        x_angiography_train, x_angiography_valid, x_angiography_test = _split_x_set(x_angiography_curr, n_train,
                                                                                    n_valid)
        x_structure_train, x_structure_valid, x_structure_test = _split_x_set(x_structure_curr, n_train, n_valid)
        x_bscan_train, x_bscan_valid, x_bscan_test = _split_x_set(x_bscan_curr, n_train, n_valid)
        x_bscan3d_train, x_bscan3d_valid, x_bscan3d_test = _split_x_set(x_bscan3d_curr, n_train, n_valid)


        x_train = [x_angiography_train, x_structure_train, x_bscan_train, x_bscan3d_train]
        x_valid = [x_angiography_valid, x_structure_valid, x_bscan_valid, x_bscan3d_valid]
        x_test = [x_angiography_test, x_structure_test, x_bscan_test, x_bscan3d_test]

        y_train = y_curr[: n_train]
        y_valid = y_curr[n_train: n_train + n_valid]
        y_test = y_curr[n_train + n_valid:]

        if len(np.unique(y_train)) == cfg.num_classes and len(np.unique(y_valid)) == cfg.num_classes and \
                len(np.unique(y_test)) == cfg.num_classes:
            break

        # else count how many times have we tried and break if failed attempt
        n_iter += 1
        if n_iter > 200:
            raise Exception("No valid splitting possible, check dataset and configuration")

    # convert the labels to onehot encoding if multi-class
    if not cfg.binary_class:
        y_train = to_categorical(y_train, num_classes=cfg.num_classes)
        y_valid = to_categorical(y_valid, num_classes=cfg.num_classes)
        y_test = to_categorical(y_test, num_classes=cfg.num_classes)

    cfg.sample_size = [x_angiography_train.shape[1:], x_bscan_train.shape[1:], x_bscan3d_train.shape[1:]]

    vec_idx_absolute = np.arange(0, x_angiography.shape[0])
    vec_idx_absolute = vec_idx_absolute[idx_permutation]
    vec_idx_absolute_train = vec_idx_absolute[: n_train]
    vec_idx_absolute_valid = vec_idx_absolute[n_train: n_train + n_valid]
    vec_idx_absolute_test = vec_idx_absolute[n_train + n_valid:]

    cfg.vec_idx_absolute = [vec_idx_absolute_train, vec_idx_absolute_valid, vec_idx_absolute_test]

    Xs = [x_train, x_valid, x_test]
    ys = [y_train, y_valid, y_test]

    return Xs, ys


def _split_data_unbalanced_cv(x_angiography, x_structure, x_bscan, x_bscan3d, y, cfg):
    """
    Unbalanced splitting of training, validation and test sets for cross validation mode

    :param x_angiography: numpy array in the form (n_sample, width, height, num_octa, 1)
    :param x_structure: numpy array in the form (n_sample, width, height, num_octa, 1)
    :param x_bscan: numpy array in the form (n_sample, width, height, 1)
    :param y: numpy array in the form (n_sample)
    :param cfg: object holding all the training parameters

    :return:
    """

    n_iter = 0
    cfg.sample_size = [x_angiography.shape[1:], x_bscan.shape[1:], x_bscan3d[1:]]
    while True:
        idx_orig = np.arange(0, x_angiography.shape[0], 1)
        idx_permutation = np.random.permutation(x_angiography.shape[0])
        vec_idx_test = np.array_split(idx_permutation, cfg.num_cv_fold)

        n_train = int(np.ceil(len(idx_permutation) * cfg.per_train))
        vec_check_fold = []
        skip_to_next_loop = False

        n_iter += 1
        if n_iter > 200:
            raise Exception("No valid splitting possible, check dataset and configuration")

        # list of lists holding the data from all folds
        vec_Xs = []
        vec_ys = []

        # list of list holding the absolute indices of all subjects
        vec_idx_absolute = []

        # loop through the folds
        for i in range(cfg.num_cv_fold):
            # extract the test set first
            idx_test_curr = vec_idx_test[i]
            x_angiography_test_curr = x_angiography[idx_test_curr, ...]
            x_structure_test_curr = x_structure[idx_test_curr, ...]
            x_bscan_test_curr = x_bscan[idx_test_curr, ...]
            x_bscan3d_test_curr = x_bscan3d[idx_test_curr, ...]
            y_test_curr = y[idx_test_curr]

            # obtain the indices for training and validation
            idx_train_valid_sorted = np.setdiff1d(idx_orig, idx_test_curr)
            idx_train_valid_permutation = np.random.permutation(idx_train_valid_sorted.shape[0])
            idx_train_valid = idx_train_valid_sorted[idx_train_valid_permutation]

            # extract the train and valid sets
            idx_train_curr = idx_train_valid[:n_train]
            idx_valid_curr = idx_train_valid[n_train:]

            x_angiography_train_curr = x_angiography[idx_train_curr, ...]
            x_structure_train_curr = x_structure[idx_train_curr, ...]
            x_bscan_train_curr = x_bscan[idx_train_curr, ...]
            x_bscan3d_train_curr = x_bscan3d[idx_train_curr, ...]
            y_train_curr = y[idx_train_curr]

            x_angiography_valid_curr = x_angiography[idx_valid_curr, ...]
            x_structure_valid_curr = x_structure[idx_valid_curr, ...]
            x_bscan_valid_curr = x_bscan[idx_valid_curr, ...]
            x_bscan3d_valid_curr = x_bscan3d[idx_valid_curr, ...]
            y_valid_curr = y[idx_valid_curr]

            # test if all classes are present in all three sets
            if len(np.unique(y_train_curr)) == cfg.num_classes and len(np.unique(y_valid_curr)) == cfg.num_classes \
                    and len(np.unique(y_test_curr)) == cfg.num_classes:
                vec_check_fold.append(True)
            else:
                skip_to_next_loop = True
                break

            x_train_curr = [x_angiography_train_curr, x_structure_train_curr, x_bscan_train_curr, x_bscan3d_train_curr]
            x_valid_curr = [x_angiography_valid_curr, x_structure_valid_curr, x_bscan_valid_curr, x_bscan3d_valid_curr]
            x_test_curr = [x_angiography_test_curr, x_structure_test_curr, x_bscan_test_curr, x_bscan3d_test_curr]

            if not cfg.binary_class:
                y_train_curr = to_categorical(y_train_curr, num_classes=cfg.num_classes)
                y_valid_curr = to_categorical(y_valid_curr, num_classes=cfg.num_classes)
                y_test_curr = to_categorical(y_test_curr, num_classes=cfg.num_classes)

            # package all the data from the current fold
            Xs_curr = [x_train_curr, x_valid_curr, x_test_curr]
            ys_curr = [y_train_curr, y_valid_curr, y_test_curr]

            # obtain the absolute indices
            vec_idx_absolute_train = idx_train_curr
            vec_idx_absolute_valid = idx_valid_curr
            vec_idx_absolute_test_curr = idx_test_curr

            # append everything to the list
            vec_Xs.append(Xs_curr)
            vec_ys.append(ys_curr)

            vec_idx_absolute.append([vec_idx_absolute_train, vec_idx_absolute_valid, vec_idx_absolute_test_curr])

        if skip_to_next_loop:
            continue

        if np.all(vec_check_fold):
            break

    cfg.vec_idx_absolute = vec_idx_absolute

    return vec_Xs, vec_ys


def _split_data(x_angiography, x_structure, x_bscan, x_bscan3d, y, cfg):
    # split into train, validation and test
    n_train = int(np.ceil(x_angiography.shape[0] * cfg.per_train))
    n_valid = int(np.floor(x_angiography.shape[0] * cfg.per_valid))

    vec_ls = np.array_split(np.arange(n_train), cfg.num_classes)
    cfg.train_split = []
    for ls in vec_ls:
        cfg.train_split.append(ls.shape[0])

    vec_idx_class0 = np.arange(x_angiography.shape[0])[y == 0]
    vec_idx_class0_permutation = np.random.permutation(vec_idx_class0.shape[0])
    vec_idx_class0 = vec_idx_class0[vec_idx_class0_permutation]
    vec_idx_class0_train = vec_idx_class0[:cfg.train_split[0]]

    vec_idx_class1 = np.arange(x_angiography.shape[0])[y == 1]
    vec_idx_class1_permutation = np.random.permutation(vec_idx_class1.shape[0])
    vec_idx_class1 = vec_idx_class1[vec_idx_class1_permutation]
    vec_idx_class1_train = vec_idx_class1[:cfg.train_split[1]]

    if not cfg.binary_class:
        vec_idx_class2 = np.arange(x_angiography.shape[0])[y == 2]
        vec_idx_class2_permutation = np.random.permutation(vec_idx_class2.shape[0])
        vec_idx_class2 = vec_idx_class2[vec_idx_class2_permutation]
        vec_idx_class2_train = vec_idx_class2[:cfg.train_split[2]]

        x_angiography_train = np.concatenate((x_angiography[vec_idx_class0_train, :, :, :, :],
                                              x_angiography[vec_idx_class1_train, :, :, :, :],
                                              x_angiography[vec_idx_class2_train, :, :, :, :]), axis=0)

        x_structure_train = np.concatenate((x_structure[vec_idx_class0_train, :, :, :, :],
                                            x_structure[vec_idx_class1_train, :, :, :, :],
                                            x_structure[vec_idx_class2_train, :, :, :, :]), axis=0)

        x_bscan_train = np.concatenate((x_bscan[vec_idx_class0_train, :, :, :],
                                        x_bscan[vec_idx_class1_train, :, :, :],
                                        x_bscan[vec_idx_class2_train, :, :, :]), axis=0)

        x_bscan3d_train = np.concatenate((x_bscan3d[vec_idx_class0_train, :, :, :, :],
                                            x_bscan3d[vec_idx_class1_train, :, :, :, :],
                                            x_bscan3d[vec_idx_class2_train, :, :, :, :]), axis=0)

        y_train = np.concatenate((y[vec_idx_class0_train],
                                  y[vec_idx_class1_train],
                                  y[vec_idx_class2_train]), axis=0)

    else:
        x_angiography_train = np.concatenate((x_angiography[vec_idx_class0_train, :, :, :, :],
                                              x_angiography[vec_idx_class1_train, :, :, :, :]), axis=0)

        x_structure_train = np.concatenate((x_structure[vec_idx_class0_train, :, :, :, :],
                                            x_structure[vec_idx_class1_train, :, :, :, :]), axis=0)

        x_bscan_train = np.concatenate((x_bscan[vec_idx_class0_train, :, :, :],
                                        x_bscan[vec_idx_class1_train, :, :, :]), axis=0)
        
        x_bscan3d_train = np.concatenate((x_bscan3d[vec_idx_class0_train, :, :, :, :],
                                            x_bscan3d[vec_idx_class1_train, :, :, :, :]), axis=0)


        y_train = np.concatenate((y[vec_idx_class0_train],
                                  y[vec_idx_class1_train]), axis=0)

    # shuffle within the training set again...
    vec_idx_train = np.arange(y_train.shape[0])
    vec_idx_train_permutation = np.random.permutation(vec_idx_train.shape[0])

    x_angiography_train = x_angiography_train[vec_idx_train_permutation, :, :, :, :]
    x_structure_train = x_structure_train[vec_idx_train_permutation, :, :, :, :]
    x_bscan_train = x_bscan_train[vec_idx_train_permutation, :, :, :]
    x_bscan3d_train = x_bscan3d_train[vec_idx_train_permutation, :, :, :, :]

    y_train = y_train[vec_idx_train_permutation]
    vec_idx_train = vec_idx_train[vec_idx_train_permutation]

    # get the absolute index wrt the indexing in x_angiography
    if not cfg.binary_class:
        vec_idx_absolute_train = np.concatenate((vec_idx_class0_train, vec_idx_class1_train, vec_idx_class2_train),
                                                axis=0)
    else:
        vec_idx_absolute_train = np.concatenate((vec_idx_class0_train, vec_idx_class1_train), axis=0)

    vec_idx_absolute_train = vec_idx_absolute_train[vec_idx_train]

    # Now generate the validation and test sets
    # TODO: right now just concatenate everything left and then split... might be a better way out there
    n_iter = 0
    while True:
        vec_idx_class0_valid_test = vec_idx_class0[cfg.train_split[0]:]
        vec_idx_class1_valid_test = vec_idx_class1[cfg.train_split[1]:]

        if not cfg.binary_class:
            vec_idx_class2_valid_test = vec_idx_class2[cfg.train_split[2]:]
            vec_idx_absolute_valid_test = np.concatenate((vec_idx_class0_valid_test, vec_idx_class1_valid_test,
                                                          vec_idx_class2_valid_test), axis=0)
            y_valid_test = np.concatenate((y[vec_idx_class0_valid_test], y[vec_idx_class1_valid_test],
                                           y[vec_idx_class2_valid_test]), axis=0)

        else:
            vec_idx_absolute_valid_test = np.concatenate((vec_idx_class0_valid_test, vec_idx_class1_valid_test), axis=0)
            y_valid_test = np.concatenate((y[vec_idx_class0_valid_test], y[vec_idx_class1_valid_test]), axis=0)

        # get a permutation and start permutating
        vec_idx_valid_test_permutation = np.random.permutation(y_valid_test.shape[0])
        vec_idx_absolute_valid_test = vec_idx_absolute_valid_test[vec_idx_valid_test_permutation]
        y_valid_test = y_valid_test[vec_idx_valid_test_permutation]

        # now split into validation and test
        vec_idx_absolute_valid = vec_idx_absolute_valid_test[:n_valid]
        y_valid = y_valid_test[:n_valid]

        vec_idx_absolute_test = vec_idx_absolute_valid_test[n_valid:]
        y_test = y_valid_test[n_valid:]

        # if there are all three labels in both sets we are done
        if len(np.unique(y_valid)) == cfg.num_classes and len(np.unique(y_test)) == cfg.num_classes:
            break

        # else count how many times have we tried and break if failed attempt
        n_iter += 1
        if n_iter > 200:
            raise Exception("No valid splitting possible, check dataset and configuration")

    x_angiography_valid = x_angiography[vec_idx_absolute_valid, :, :, :, :]
    x_structure_valid = x_structure[vec_idx_absolute_valid, :, :, :, :]
    x_bscan_valid = x_bscan[vec_idx_absolute_valid, :, :, :]
    x_bscan3d_valid = x_bscan3d[vec_idx_absolute_valid, :, :, :, :]

    y_valid_alt = y[vec_idx_absolute_valid]
    if not np.allclose(y_valid, y_valid_alt):
        raise Exception("Indexing mismatch")

    x_angiography_test = x_angiography[vec_idx_absolute_test, :, :, :, :]
    x_structure_test = x_structure[vec_idx_absolute_test, :, :, :, :]
    x_bscan_test = x_bscan[vec_idx_absolute_test, :, :, :]
    x_bscan3d_test = x_bscan3d[vec_idx_absolute_test, :, :, :, :]

    y_test_alt = y[vec_idx_absolute_test]
    if not np.allclose(y_test, y_test_alt):
        raise Exception("Indexing mismatch")

    cfg.vec_idx_absolute = [vec_idx_absolute_train, vec_idx_absolute_valid, vec_idx_absolute_test]

    # convert the labels to onehot encoding
    if not cfg.binary_class:
        y_train = to_categorical(y_train, num_classes=cfg.num_classes)
        y_valid = to_categorical(y_valid, num_classes=cfg.num_classes)
        y_test = to_categorical(y_test, num_classes=cfg.num_classes)

    x_train = [x_angiography_train, x_structure_train, x_bscan_train, x_bscan3d_train]
    x_valid = [x_angiography_valid, x_structure_valid, x_bscan_valid, x_bscan3d_valid]
    x_test = [x_angiography_test, x_structure_test, x_bscan_test, x_bscan3d_test]

    cfg.sample_size = [x_angiography_train.shape[1:], x_bscan_train.shape[1:], x_bscan3d_train[1:]]

    return [x_train, x_valid, x_test], [y_train, y_valid, y_test]


def _split_x_set(x, n_train, n_valid):
    if len(x.shape) == 5:
        x_train = x[: n_train, :, :, :, :]
        x_valid = x[n_train: n_train + n_valid, :, :, :, :]
        x_test = x[n_train + n_valid:, :, :, :, :]
    else:
        x_train = x[: n_train, :, :, :]
        x_valid = x[n_train: n_train + n_valid, :, :, :]
        x_test = x[n_train + n_valid:, :, :, :]

    return x_train, x_valid, x_test


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
        if cfg.str_feature not in cfg.vec_all_str_feature:
            raise Exception('Invalid feature label provided')

        vec_str_patient_id, vec_OD_feature, vec_OS_feature = load_csv_params(cfg)

        X, y, vec_str_patients, vec_out_csv_idx = load_all_data_csv(vec_idx_patient, vec_str_patient_id, vec_OD_feature,
                                                                    vec_OS_feature, cfg)

        cfg.vec_str_patients = vec_str_patients
        cfg.vec_out_csv_idx = vec_out_csv_idx
        if not cfg.binary_class:
            cfg.y_unique_label = np.unique(y)

            # check if there are only two labels present, which is the case for many features
            if len(np.unique(y)) == 2 and cfg.str_feature != 'disease':
                cfg.num_classes = 2
                cfg.binary_class = True

                vec_str_labels_temp = []
                for i in range(cfg.num_classes):
                    vec_str_labels_temp.append(cfg.vec_str_labels[np.unique(y)[i]])
                cfg.vec_str_labels = vec_str_labels_temp

                # correct for labels where there are skips
                y_temp = y.copy()
                if not np.all(np.unique(y) == np.arange(0, cfg.num_classes)):
                    for i in range(cfg.num_classes):
                        y_temp[y_temp == np.unique(y)[i]] = np.arange(0, cfg.num_classes)[i]
                y = y_temp

            elif len(np.unique(y)) == 2 and cfg.str_feature == 'disease':
                raise Exception('There should be three disease labels')

            elif len(np.unique(y)) == 4:
                raise Exception('Too many labels')
        else:
            cfg.y_unique_label = np.arange(0, 2, 1)
            y_temp = y.copy()
            if cfg.binary_mode == 0:
                idx_label_0 = y_temp == 0
                idx_label_1 = y_temp == 1
                cfg.vec_str_labels = ['Normal', 'NNV AMD']

            elif cfg.binary_mode == 1:
                idx_label_0 = y_temp == 0
                idx_label_1 = y_temp == 2
                cfg.vec_str_labels = ['Normal', 'NV AMD']

            elif cfg.binary_mode == 2:
                idx_label_0 = y_temp == 1
                idx_label_1 = y_temp == 2
                cfg.vec_str_labels = ['NNV AMD', 'NV AMD']

            X_angio_label0 = X[0][idx_label_0, ...]
            X_struct_label0 = X[1][idx_label_0, ...]
            X_bscan_label0 = X[2][idx_label_0, ...]
            X_bscan3d_label0 = X[3][idx_label_0, ...]
            y_label0 = np.zeros_like(y[idx_label_0])

            X_angio_label1 = X[0][idx_label_1, ...]
            X_struct_label1 = X[1][idx_label_1, ...]
            X_bscan_label1 = X[2][idx_label_1, ...]
            X_bscan3d_label1 = X[3][idx_label_1, ...]
            y_label1 = np.ones_like(y[idx_label_1])

            idx_full = np.arange(0, len(cfg.vec_str_patients), 1)
            idx_binary_label0 = idx_full[idx_label_0]
            idx_binary_label1 = idx_full[idx_label_1]
            idx_binary = np.concatenate([idx_binary_label0, idx_binary_label1])
            idx_binary_sort = np.argsort(idx_binary)
            idx_binary = idx_binary[idx_binary_sort]

            X_angio = np.concatenate([X_angio_label0, X_angio_label1])
            X_angio = X_angio[idx_binary_sort, ...]
            X_struct = np.concatenate([X_struct_label0, X_struct_label1])
            X_struct = X_struct[idx_binary_sort, ...]
            X_bscan = np.concatenate([X_bscan_label0, X_bscan_label1])
            X_bscan = X_bscan[idx_binary_sort, ...]
            X_bscan3d = np.concatenate([X_bscan3d_label0, X_bscan3d_label1])
            X_bscan3d = X_bscan3d[idx_binary_sort, ...]
            X = [X_angio, X_struct, X_bscan, X_bscan3d]

            y = np.concatenate([y_label0, y_label1])
            y = y[idx_binary_sort]

            vec_str_patients_temp = []
            vec_out_csv_idx_temp = []
            for i in range(len(cfg.vec_str_patients)):
                if i in idx_binary:
                    vec_str_patients_temp.append(cfg.vec_str_patients[i])
                    vec_out_csv_idx_temp.append(cfg.vec_out_csv_idx)

            cfg.vec_str_patients = vec_str_patients_temp
            cfg.vec_out_csv_idx = vec_out_csv_idx_temp

    else:
        raise Exception('Undefined load mode')

    return X, y


def load_all_data_csv(vec_idx, vec_str_patient_id, vec_OD_feature, vec_OS_feature, cfg):
    """
    Functional wrapper for loading data from all patients using function below

    :param vec_idx: list in the form of [start_idx, end_idx]
    :param cfg: configuration file set by the user
    :return:
    """

    x, y, vec_str_patients, vec_out_csv_idx = _load_all_data_csv(vec_idx, vec_str_patient_id, 
                                                                 vec_OD_feature, vec_OS_feature, cfg.d_data,
                                                                 cfg.downscale_size, cfg.downscale_size_bscan,
                                                                 cfg.crop_size, cfg.num_octa,
                                                                 cfg.str_angiography, cfg.str_structure, cfg.str_bscan,
                                                                 cfg.vec_str_layer, cfg.vec_str_layer_bscan3d, 
                                                                 cfg.str_bscan_layer, cfg.dict_layer_order, 
                                                                 cfg.dict_layer_order_bscan3d, 
                                                                 cfg.vec_csv_col)

    return x, y, vec_str_patients, vec_out_csv_idx


def _load_all_data_csv(vec_idx, vec_str_patient_id, vec_OD_feature, vec_OS_feature,
                       d_data, downscale_size, downscale_size_bscan, crop_size, num_octa, str_angiography, str_structure,
                       str_bscan, vec_str_layer, vec_str_layer_bscan3d,
                       str_bscan_layer, dict_layer_order, dict_layer_order_bscan3d, vec_csv_col):

    """
    Load all data from all patients without assigning the class label yet

    :param vec_idx: list in the form of [start_idx, end_idx]
    :param vec_str_patient_id:
    :param vec_OD_feature:
    :param vec_OS_feature:
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
    :param vec_csv_col: list of all the indices of relevant columns in the original csv file

    :return: a tuple in the form [x_angiography, x_structure, x_bscan], vec_str_patient, where each of x_class
    contains images from a single type of image and vec_str_patient would correspond to absolute
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

    # create a list of all valid entries in the csv file
    vec_out_csv_idx = []
    idx_col_OD_feature = vec_csv_col[-2]
    idx_col_OS_feature = vec_csv_col[-1]

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
                if str_eye[j] == 'OD':
                    y_curr = vec_OD_feature[rel_idx_patient_id]
                    vec_out_csv_idx.append([rel_idx_patient_id, idx_col_OD_feature])
                elif str_eye[j] == 'OS':
                    y_curr = vec_OS_feature[rel_idx_patient_id]
                    vec_out_csv_idx.append([rel_idx_patient_id, idx_col_OS_feature])
                else:
                    raise Exception('Invalid eye label encountered')

                if np.isnan(y_curr):
                    raise Exception("Label shouldn't be NaN")

                x_angiography.append(packed_x_curr[j][0])
                x_structure.append(packed_x_curr[j][1])
                x_bscan.append(packed_x_curr[j][2])
                x_bscan3d.append(packed_x_curr[j][3])

                y.append(int(y_curr))

                # append to list of patients
                str_patient = "Patient {}/{}".format(vec_full_idx[i], str_eye[j])
                vec_str_patient.append(str_patient)

        else:
            if str_eye == 'OD':
                y_curr = vec_OD_feature[rel_idx_patient_id]
                vec_out_csv_idx.append([rel_idx_patient_id, idx_col_OD_feature])
            elif str_eye == 'OS':
                y_curr = vec_OS_feature[rel_idx_patient_id]
                vec_out_csv_idx.append([rel_idx_patient_id, idx_col_OS_feature])
            else:
                raise Exception('Invalid eye label encountered')

            if np.isnan(y_curr):
                raise Exception("Label shouldn't be NaN")

            x_angiography.append(packed_x_curr[0])
            x_structure.append(packed_x_curr[1])
            x_bscan.append(packed_x_curr[2])
            x_bscan3d.append(packed_x_curr[3])

            y.append(int(y_curr))

            # append to list of patients
            str_patient = "Patient {}/{}".format(vec_full_idx[i], str_eye)
            vec_str_patient.append(str_patient)

    x_angiography = np.stack(x_angiography, axis=0)
    x_structure = np.stack(x_structure, axis=0)
    x_bscan = np.stack(x_bscan, axis=0)
    x_bscan3d = np.stack(x_bscan3d, axis=0)

    y = np.stack(y, axis=0)
    vec_out_csv_idx = vec_out_csv_idx

    return [x_angiography, x_structure, x_bscan, x_bscan3d], y, vec_str_patient, vec_out_csv_idx


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
