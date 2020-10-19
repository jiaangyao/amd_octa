import glob
import re

import numpy as np
from skimage import io, transform, color
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

from context_management import temp_seed


def preprocess(vec_idx_healthy, vec_idx_dry_amd, vec_idx_cnv, cfg):
    print('Loading data from normal patients')
    x_healthy, y_healthy = load_data(vec_idx_healthy, cfg.str_healthy, cfg.label_healthy,
                                     cfg.d_data, cfg.downscale_size, cfg.num_octa, cfg.str_angiography,
                                     cfg.str_structural, cfg.str_bscan, cfg.vec_str_layer, cfg.str_bscan_layer,
                                     cfg.dict_layer_order)

    print('\nLoading data from dry AMD patients')
    x_dry_amd, y_dry_amd = load_data(vec_idx_dry_amd, cfg.str_dry_amd, cfg.label_dry_amd,
                                     cfg.d_data, cfg.downscale_size, cfg.num_octa, cfg.str_angiography,
                                     cfg.str_structural, cfg.str_bscan, cfg.vec_str_layer, cfg.str_bscan_layer,
                                     cfg.dict_layer_order)

    print('\nLoading data from CNV patients')
    x_cnv, y_cnv = load_data(vec_idx_cnv, cfg.str_cnv, cfg.label_cnv,
                             cfg.d_data, cfg.downscale_size, cfg.num_octa, cfg.str_angiography,
                             cfg.str_structural, cfg.str_bscan, cfg.vec_str_layer, cfg.str_bscan_layer,
                             cfg.dict_layer_order)

    cfg.n_healthy = x_healthy[0].shape[0]
    cfg.n_dry_amd = x_dry_amd[0].shape[0]
    cfg.n_cnv = x_cnv[0].shape[0]

    # unpack once more
    x_angiography = np.concatenate((x_healthy[0], x_dry_amd[0], x_cnv[0]), axis=0)
    x_structure = np.concatenate((x_healthy[1], x_dry_amd[1], x_cnv[1]), axis=0)
    x_bscan = np.concatenate((x_healthy[2], x_dry_amd[2], x_cnv[2]), axis=0)
    y = np.concatenate((y_healthy, y_dry_amd, y_cnv), axis=0)

    if cfg.oversample:
        if cfg.oversample_method == 'smote':
            from imblearn.over_sampling import SMOTE

            # TODO: very bad form... fix this later

            x_angiography_rs = x_angiography.reshape(x_angiography.shape[0], -1)
            x_structure_rs = x_structure.reshape(x_structure.shape[0], -1)
            x_bscan_rs = x_bscan.reshape(x_bscan.shape[0], -1)

            sm = SMOTE()
            x_angiography_rs, y_rs = sm.fit_resample(x_angiography_rs, y)
            x_structure_rs, y_rs_alt = sm.fit_resample(x_structure_rs, y)
            x_bscan_rs, y_rs_alt_alt = sm.fit_resample(x_bscan_rs, y)

            angio_shape = [x_angiography_rs.shape[0]]
            angio_shape.extend(list(x_angiography.shape[1:]))

            structure_shape = [x_structure_rs.shape[0]]
            structure_shape.extend(list(x_structure.shape[1:]))

            bscan_shape = [x_bscan_rs.shape[0]]
            bscan_shape.extend(list(x_bscan.shape[1:]))

            x_angiography_rs = x_angiography_rs.reshape(angio_shape)
            x_structure_rs = x_structure_rs.reshape(structure_shape)
            x_bscan_rs = x_bscan_rs.reshape(bscan_shape)

            if not (np.allclose(y_rs, y_rs_alt) and np.allclose(y_rs, y_rs_alt_alt)):
                raise Exception("Issues with SMOTE")

        elif cfg.oversample_method == 'random':
            pass

    # clearing the variables for memory purposes
    x_healthy = []
    y_healthy = []

    x_dry_amd = []
    y_dry_amd = []

    x_cnv = []
    y_cnv = []

    if not cfg.oversample:
        if cfg.use_random_seed:
            with temp_seed(cfg.random_seed):
                Xs, ys = _split_data(x_angiography, x_structure, x_bscan, y, cfg)
        else:
            Xs, ys = _split_data(x_angiography, x_structure, x_bscan, y, cfg)
    else:
        if cfg.use_random_seed:
            with temp_seed(cfg.random_seed):
                Xs, ys = _split_data(x_angiography_rs, x_structure_rs, x_bscan_rs, y_rs, cfg)
        else:
            Xs, ys = _split_data(x_angiography_rs, x_structure_rs, x_bscan_rs, y_rs, cfg)

    return Xs, ys


def _split_data(x_angiography, x_structure, x_bscan, y, cfg):

    # split into train, validation and test
    n_train = int(np.ceil(x_angiography.shape[0] * cfg.per_train))
    n_valid = int(np.floor(x_angiography.shape[0] * cfg.per_valid))

    vec_ls = np.array_split(np.arange(n_train), cfg.num_classes)
    cfg.train_split = [vec_ls[0].shape[0], vec_ls[1].shape[0], vec_ls[1].shape[0]]

    # TODO: Hard coding this for now but ugly....
    vec_idx_class0 = np.arange(x_angiography.shape[0])[y == 0]
    vec_idx_class1 = np.arange(x_angiography.shape[0])[y == 1]
    vec_idx_class2 = np.arange(x_angiography.shape[0])[y == 2]

    vec_idx_class0_permutation = np.random.permutation(vec_idx_class0.shape[0])
    vec_idx_class1_permutation = np.random.permutation(vec_idx_class1.shape[0])
    vec_idx_class2_permutation = np.random.permutation(vec_idx_class2.shape[0])

    vec_idx_class0 = vec_idx_class0[vec_idx_class0_permutation]
    vec_idx_class1 = vec_idx_class1[vec_idx_class1_permutation]
    vec_idx_class2 = vec_idx_class2[vec_idx_class2_permutation]

    vec_idx_class0_train = vec_idx_class0[:cfg.train_split[0]]
    vec_idx_class1_train = vec_idx_class1[:cfg.train_split[1]]
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

    y_train = np.concatenate((y[vec_idx_class0_train],
                              y[vec_idx_class1_train],
                              y[vec_idx_class2_train]), axis=0)

    vec_idx_train = np.arange(y_train.shape[0])
    vec_idx_train_permutation = np.random.permutation(vec_idx_train.shape[0])

    # shuffle within the training set again...
    x_angiography_train = x_angiography_train[vec_idx_train_permutation, :, :, :, :]
    x_structure_train = x_structure_train[vec_idx_train_permutation, :, :, :, :]
    x_bscan_train = x_bscan_train[vec_idx_train_permutation, :, :, :]
    y_train = y_train[vec_idx_train_permutation]
    vec_idx_train = vec_idx_train[vec_idx_train_permutation]

    # get the absolute index wrt the indexing in x_angiography
    vec_idx_absolute_train = np.concatenate((vec_idx_class0_train, vec_idx_class1_train, vec_idx_class2_train), axis=0)
    vec_idx_absolute_train = vec_idx_absolute_train[vec_idx_train]

    # Now generate the validation and test sets
    # TODO: right now just concatenate everything left and then split
    vec_idx_class0_valid_test = vec_idx_class0[cfg.train_split[0]:]
    vec_idx_class1_valid_test = vec_idx_class1[cfg.train_split[1]:]
    vec_idx_class2_valid_test = vec_idx_class2[cfg.train_split[2]:]

    while True:
        vec_idx_absolute_valid_test = np.concatenate((vec_idx_class0_valid_test, vec_idx_class1_valid_test,
                                                      vec_idx_class2_valid_test), axis=0)
        y_valid_test = np.concatenate((y[vec_idx_class0_valid_test], y[vec_idx_class1_valid_test],
                                       y[vec_idx_class2_valid_test]), axis=0)

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
        if len(np.unique(y_valid)) == 3 and len(np.unique(y_test)) == 3:
            break

    x_angiography_valid = x_angiography[vec_idx_absolute_valid, :, :, :, :]
    x_structure_valid = x_structure[vec_idx_absolute_valid, :, :, :, :]
    x_bscan_valid = x_bscan[vec_idx_absolute_valid, :, :, :]
    y_valid_alt = y[vec_idx_absolute_valid]
    if not np.allclose(y_valid, y_valid_alt):
        raise Exception("Indexing mismatch")

    x_angiography_test = x_angiography[vec_idx_absolute_test, :, :, :, :]
    x_structure_test = x_structure[vec_idx_absolute_test, :, :, :, :]
    x_bscan_test = x_bscan[vec_idx_absolute_test, :, :, :]
    y_test_alt = y[vec_idx_absolute_test]
    if not np.allclose(y_test, y_test_alt):
        raise Exception("Indexing mismatch")

    cfg.vec_idx_absolute = [vec_idx_absolute_train, vec_idx_absolute_valid, vec_idx_absolute_test]

    # convert the labels to onehot encoding
    y_train = to_categorical(y_train, num_classes=cfg.num_classes)
    y_valid = to_categorical(y_valid, num_classes=cfg.num_classes)
    y_test = to_categorical(y_test, num_classes=cfg.num_classes)

    x_train = [x_angiography_train, x_structure_train, x_bscan_train]
    x_valid = [x_angiography_valid, x_structure_valid, x_bscan_valid]
    x_test = [x_angiography_test, x_structure_test, x_bscan_test]

    cfg.sample_size = [x_angiography_train.shape[1:], x_bscan_train.shape[1:]]

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


def load_data(vec_idx, str_class, label_class, d_data, downscale_size, num_octa, str_angiography, str_structural,
              str_bscan, vec_str_layer, str_bscan_layer, dict_layer_order):
    """
    Load data of a specific class

    :param vec_idx: list in the form of [start_idx, end_idx]
    :param str_class: name of this class, e.g. normalPatient
    :param label_class: label assigned to this class, e.g. 0
    :param pathlib.Path d_data: directory to the data
    :param list downscale_size: desired shape after downscaling the images, e.g. [350, 350]
    :param num_octa:
    :param str_angiography:
    :param str_structural:
    :param str_bscan:
    :param vec_str_layer:
    :param str_bscan_layer:
    :param dict_layer_order:

    :return:
    """

    # note that when loss is binary cross entropy then no need for onehot of y
    # but later when switching to multiclass and softmax then want to use either categorical cross entropy and onehot
    # or sparse cross entropy
    # TODO: change the label once we swtich to three classes

    # create the empty lists for holding the variables
    x_angiography = []
    x_structure = []
    x_bscan = []

    y = []

    # create a list of all possible indices
    vec_full_idx = np.arange(vec_idx[0], vec_idx[1] + 1, 1)

    # Loop through all the runs
    for i in range(len(vec_full_idx)):
        vec_f_image = glob.glob(str(d_data / str_class / '[Pp]atient {}'.format(vec_full_idx[i]) / '**' / '*.bmp'),
                                recursive=True)

        if len(vec_f_image) == 0:
            vec_f_image = glob.glob(str(d_data / str_class / '[Pp]atient {} - *'.format(vec_full_idx[i]) / '**' / '*.bmp'),
                                    recursive=True)

        if vec_f_image:
            print("Loading data from patient {}".format(vec_full_idx[i]))

        else:
            # print("Data not available for patient {}, skipping...".format(vec_full_idx[i]))
            continue

        packed_x_curr = package_data(vec_f_image, downscale_size, num_octa, str_angiography, str_structural,
                                     str_bscan, vec_str_layer, str_bscan_layer, dict_layer_order)

        if packed_x_curr is None:
            continue

        # now unpack the data
        if len(packed_x_curr) == 2:
            for j in range(len(packed_x_curr)):
                x_angiography.append(packed_x_curr[j][0])
                x_structure.append(packed_x_curr[j][1])
                x_bscan.append(packed_x_curr[j][2])

                # append the class label also
                y.append(label_class)

        else:
            x_angiography.append(packed_x_curr[0])
            x_structure.append(packed_x_curr[1])
            x_bscan.append(packed_x_curr[2])

            # append the class label also
            y.append(label_class)

    x_angiography = np.stack(x_angiography, axis=0)
    x_structure = np.stack(x_structure, axis=0)
    x_bscan = np.stack(x_bscan, axis=0)

    return [x_angiography, x_structure, x_bscan], y


def package_data(vec_f_image, downscale_size, num_octa, str_angiography, str_structural, str_bscan,
                 vec_str_layer, str_bscan_layer, dict_layer_order):
    """

    :param vec_f_image:
    :param downscale_size:
    :param num_octa:
    :param str_angiography:
    :param str_structural:
    :param str_bscan:
    :param vec_str_layer:
    :param str_bscan_layer:
    :param dict_layer_order:
    :return:
    """

    # Test if the dataset has data from both eyes
    if any("OD" in s for s in vec_f_image) & any("OS" in s for s in vec_f_image):

        vec_f_image_OD = [s for s in vec_f_image if "OD" in s]
        vec_f_image_OS = [s for s in vec_f_image if "OS" in s]

        x_curr_OD = _form_cubes(vec_f_image_OD, num_octa, downscale_size, str_angiography, str_structural,
                                str_bscan, vec_str_layer, str_bscan_layer, dict_layer_order)

        x_curr_OS = _form_cubes(vec_f_image_OS, num_octa, downscale_size, str_angiography, str_structural,
                                str_bscan, vec_str_layer, str_bscan_layer, dict_layer_order)

        # Figure out if any of the single eye data is none
        if x_curr_OD is not None and x_curr_OS is not None:
            packed_x_curr = [x_curr_OD, x_curr_OS]
        elif x_curr_OD is None and x_curr_OS is not None:
            packed_x_curr = x_curr_OS
        elif x_curr_OD is not None and x_curr_OS is None:
            packed_x_curr = x_curr_OD
        else:
            packed_x_curr = None

    else:
        x_curr = _form_cubes(vec_f_image, num_octa, downscale_size, str_angiography, str_structural,
                             str_bscan, vec_str_layer, str_bscan_layer, dict_layer_order)

        packed_x_curr = x_curr

    return packed_x_curr


def _form_cubes(vec_f_image, num_octa, downscale_size, str_angiography, str_structural, str_bscan, vec_str_layer,
                str_bscan_layer, dict_layer_order):
    """

    :param vec_f_image:
    :param str_angiography:
    :param str_structural:
    :param str_bscan:
    :param vec_str_layer:
    :param str_bscan_layer:
    :param dict_layer_order:
    :return:
    """
    # TODO: this function is heavily hardcoded at this point...

    vec_vol_f_image_angiography_curr = {}
    vec_vol_f_image_structure_curr = {}
    vec_vol_f_image_bscan_curr = {}

    for f_image in vec_f_image:

        p_f_image = pathlib.Path(f_image)
        p_f_image_filename = p_f_image.name

        for str_image_type in [str_angiography, str_structural, str_bscan]:
            if str_image_type == str_bscan:
                re_pattern_bscan = '.*{} {}.bmp'.format(str_image_type, str_bscan_layer)

                re_hits = re.findall(re_pattern_bscan, p_f_image_filename, re.I)

                if re_hits:
                    vec_vol_f_image_bscan_curr[len(vec_vol_f_image_bscan_curr)] = f_image
            else:
                for str_layer in vec_str_layer:
                    re_pattern_curr = '.*{}_{}.bmp'.format(str_image_type, str_layer)

                    re_hits = re.findall(re_pattern_curr, p_f_image_filename, re.I)

                    if re_hits:
                        if str_image_type == str_angiography:
                            vec_vol_f_image_angiography_curr[dict_layer_order[str_layer]] = f_image

                        else:
                            vec_vol_f_image_structure_curr[dict_layer_order[str_layer]] = f_image

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
    vol_bscan_curr = np.zeros([downscale_size[0], downscale_size[1], 1])

    _create_np_cubes(vol_angiography_curr, vec_vol_f_image_angiography_curr, downscale_size)
    _create_np_cubes(vol_structure_curr, vec_vol_f_image_structure_curr, downscale_size)
    _create_np_cubes(vol_bscan_curr, vec_vol_f_image_bscan_curr, downscale_size)

    x_curr = [vol_angiography_curr, vol_structure_curr, vol_bscan_curr]

    return x_curr


def _load_individual_image(f_image, downscale_size):
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

    return imgResized


def _create_np_cubes(np_cube, vec_vol_f_image, downscale_size):
    if len(np_cube.shape) == 4:
        for idx_layer, p_f_image_curr_layer in vec_vol_f_image.items():
            curr_img = _load_individual_image(str(p_f_image_curr_layer), downscale_size)
            np_cube[:, :, idx_layer, :] = curr_img

    # TODO: code for B scan is ugly
    else:
        np_cube[:, :] = _load_individual_image(str(vec_vol_f_image[0]), downscale_size)

