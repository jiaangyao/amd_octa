import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

from utils.context_management import temp_seed


def prepare_data_for_train(X, y, cfg):
    # Exception detection
    if not cfg.binary_class:
        if not cfg.num_classes == 3:
            raise Exception('Three class classification specified but two classes requested')
    else:
        if not cfg.num_classes == 2:
            raise Exception('Binary classification specified but three classes requested')

    # load all data
    [x_angiography, x_structure, x_bscan, x_bscan3d] = X

    # split the data into training, validation and test set
    if not cfg.balanced:
        if cfg.use_random_seed:
            with temp_seed(cfg.random_seed):
                Xs, ys = _split_data_unbalanced(x_angiography, x_structure, x_bscan, x_bscan3d, y, cfg)
        else:
            Xs, ys = _split_data_unbalanced(x_angiography, x_structure, x_bscan, x_bscan3d, y, cfg)

        if cfg.oversample:
            if cfg.oversample_method == 'smote':
                raise NotImplementedError

            elif cfg.oversample_method == 'random':
                raise NotImplementedError

    else:
        if cfg.use_random_seed:
            with temp_seed(cfg.random_seed):
                Xs, ys = _split_data(x_angiography, x_structure, x_bscan, x_bscan3d, y, cfg)
        else:
            Xs, ys = _split_data(x_angiography, x_structure, x_bscan, x_bscan3d, y, cfg)

    return Xs, ys


def prepare_data_for_train_cv(X, y, cfg):
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
    [x_angiography, x_structure, x_bscan, x_bscan3d] = X

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
    cfg.sample_size = [x_angiography.shape[1:], x_bscan.shape[1:], x_bscan3d.shape[1:]]
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

            # sanity check
            if not x_angiography_train_curr.shape[0] == x_structure_train_curr.shape[0] == \
                   x_bscan_train_curr.shape[0] == x_bscan3d_train_curr.shape[0]:
                raise ValueError("These should be equal")
            if not x_angiography_valid_curr.shape[0] == x_structure_valid_curr.shape[0] == \
                   x_bscan_valid_curr.shape[0] == x_bscan3d_valid_curr.shape[0]:
                raise ValueError("These should be equal")
            if not x_angiography_test_curr.shape[0] == x_structure_test_curr.shape[0] == \
                   x_bscan_test_curr.shape[0] == x_bscan3d_test_curr.shape[0]:
                raise ValueError("These should be equal")

            if not cfg.binary_class:
                y_train_curr = to_categorical(y_train_curr, num_classes=cfg.num_classes)
                y_valid_curr = to_categorical(y_valid_curr, num_classes=cfg.num_classes)
                y_test_curr = to_categorical(y_test_curr, num_classes=cfg.num_classes)

            # sanity check
            if not x_angiography_train_curr.shape[0] == y_train_curr.shape[0]:
                raise ValueError("These should be equal")
            if not x_angiography_valid_curr.shape[0] == y_valid_curr.shape[0]:
                raise ValueError("These should be equal")
            if not x_angiography_test_curr.shape[0] == y_test_curr.shape[0]:
                raise ValueError("These should be equal")

            # package all the data from the current fold
            Xs_curr = [x_train_curr, x_valid_curr, x_test_curr]
            ys_curr = [y_train_curr, y_valid_curr, y_test_curr]

            # obtain the absolute indices
            vec_idx_absolute_train = idx_train_curr
            vec_idx_absolute_valid = idx_valid_curr
            vec_idx_absolute_test = idx_test_curr

            # append everything to the list
            vec_Xs.append(Xs_curr)
            vec_ys.append(ys_curr)

            vec_idx_absolute.append([vec_idx_absolute_train, vec_idx_absolute_valid, vec_idx_absolute_test])

            # sanity check
            if not vec_idx_absolute_train.shape[0] == x_angiography_train_curr.shape[0]:
                raise ValueError("These should be equal")
            if not vec_idx_absolute_valid.shape[0] == x_angiography_valid_curr.shape[0]:
                raise ValueError("These should be equal")
            if not vec_idx_absolute_test.shape[0] == x_angiography_test_curr.shape[0]:
                raise ValueError("These should be equal")

        if skip_to_next_loop:
            continue

        if np.all(vec_check_fold):
            break

    # sanity check
    vec_len_xs = []
    vec_len_ys = []
    for i in range(len(vec_Xs)):
        vec_len_xs.append(vec_Xs[i][0][0].shape[0] + vec_Xs[i][1][0].shape[0] + vec_Xs[i][2][0].shape[0])
        vec_len_ys.append(vec_ys[i][0].shape[0] + vec_ys[i][1].shape[0] + vec_ys[i][2].shape[0])

    if not np.allclose(vec_len_xs, vec_len_ys):
        raise ValueError("These should be equal")
    if not np.all(np.stack(vec_len_xs, axis=0) == vec_len_xs[0]):
        raise ValueError("These should be equal")

    # append hyperparameter to the configuration file
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

    vec_idx_class2 = None
    vec_idx_class2_train = None
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

    cfg.sample_size = [x_angiography_train.shape[1:], x_bscan_train.shape[1:], x_bscan3d_train.shape[1:]]

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