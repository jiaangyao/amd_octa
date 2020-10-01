import glob

import numpy as np
from skimage import io, transform, color
import matplotlib.pyplot as plt


def preprocess(vec_idx_run_healthy, vec_idx_dry_amd, cfg):

    X_healthy, y_healthy = load_data(vec_idx_run_healthy, cfg.str_healthy, cfg.label_healthy,
                     cfg.d_data, cfg.downscale_size)

    X_dry_amd, y_dry_amd = load_data(vec_idx_dry_amd, cfg.str_dry_amd, cfg.label_dry_amd,
                                     cfg.d_data, cfg.downscale_size)

    # quite stupid way of implementing this but okay for now..
    X = np.append(X_healthy, X_dry_amd, axis=0)
    y = np.append(y_healthy, y_dry_amd, axis=0)

    # TODO: might want to get a separate function for this
    # shuffle
    idx_permutation = np.random.permutation(X.shape[0])
    X = X[idx_permutation, :, :, :, :]
    y = y[idx_permutation]

    # split into train, validation and test
    n_train = int(np.ceil(X.shape[0] * cfg.per_train))
    n_valid = int(np.floor(X.shape[0] * cfg.per_valid))

    X_train = X[: n_train, :, :, :, :]
    y_train = y[: n_train]

    X_valid = X[n_train: n_train + n_valid, :, :, :, :]
    y_valid = y[n_train: n_train + n_valid]

    X_test = X[n_train + n_valid:, :, :, :, :]
    y_test = y[n_train + n_valid:]
    cfg.sample_size = X_train.shape[1:]

    return [X_train, X_valid, X_test], [y_train, y_valid, y_test]


def load_data(vec_idx_run, str_class, label_class, d_data, downscale_size):
    """
    Load data of a specific class

    :param vec_idx_run: list of all runs
    :param str_class: name of this class, e.g. normalPatient
    :param label_class: label assigned to this class, e.g. 0
    :param pathlib.Path d_data: directory to the data
    :param list downscale_size: desired shape after downscaling the images, e.g. [350, 350]

    :return:
    """

    # note that when loss is binary cross entropy then no need for onehot of y
    # but later when switching to multiclass and softmax then want to use either categorical cross entropy and onehot
    # or sparse cross entropy
    # TODO: change the label once we swtich to three classes

    # create the empty lists for holding the variables
    X = []
    y = []

    # Loop through all the runs
    for i in range(len(vec_idx_run)):
        vol_curr = []
        for f_image in glob.glob(str(d_data / "{}{}".format(str_class, vec_idx_run[i]) / '**' / '*.bmp'), recursive=True):
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

            vol_curr.append(imgResized)

        # stack into a numpy array of shape (height, width, depth, 1)
        vol_curr_np = np.stack(vol_curr, axis=-2)
        X.append(vol_curr_np)

        # append the class label also
        y.append(label_class)

    # TODO: right now I don't think we are going to encounter this issue but worry about masking later
    # why is masking needed: some patients don't have as many images as others
    # why is masking challenging: don't think Keras supports conv3D with masking just yet
    #   so need to write custom layer that supports masking;
    #   see https://israelg99.github.io/2017-02-27-Grayscale-PixelCNN-with-Keras/
    # alternative method: apply the time distributed layer and treat each slice separately...
    #   but that might not be what we want here...

    # TODO: delete this bit once we have same amount of images for everyone
    min_n_images = 1000
    for x in X:
        if x.shape[-2] < min_n_images:
            min_n_images = x.shape[-2]

    X_trimed = []
    for x in X:
        X_trimed.append(x[:, :, :min_n_images, :])

    # now stack these into numpy arrays
    # TODO: change below to X later
    X = np.stack(X_trimed, axis=0)
    y = np.stack(y, axis=0)

    return X, y