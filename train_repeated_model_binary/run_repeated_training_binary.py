import os
import pathlib
import copy
import time
import pickle

import numpy as np

from config.load_config import get_config
from config.config_utils import initialize_config_preproc, initialize_config_split, initialize_config_training
from preproc.preprocess import generate_labels, correct_data_label
from preproc.train_val_test_split import prepare_data_for_train
from save_load.io_funcs import save_cfg, save_mat
from modeling.model import get_model, get_callbacks
from analysis.plotting import plot_norm_conf_matrix, plot_raw_conf_matrix


def run_repeated_training_binary(cfg_in):
    # create the empty lists for holding all the data
    vec_train_acc = []
    vec_valid_acc = []
    vec_test_acc = []

    vec_y_true = []
    vec_y_pred = []
    vec_y_pred_prob = []

    mat_vec_idx_absolute_test = []

    # set the random seed for all the splits
    temp = initialize_config_split(copy.deepcopy(cfg_in))
    np.random.seed(temp.random_seed)

    # initiate the repeated training
    for i in range(cfg_in.n_repeats):
        cfg = copy.deepcopy(cfg_in)
        print("\n\nIteration: {}".format(i + 1))

        # now load the preprocessed data and the label
        f_data_handle = "preproc_data_{}_{}.pkl".format(cfg.vec_idx_patient[0], cfg.vec_idx_patient[1])
        f_data = cfg.d_preproc / f_data_handle
        with open(str(f_data), 'rb') as handle:
            X = pickle.load(handle)

        y = generate_labels(cfg.str_feature, cfg, bool_append_csv_to_cfg=True)

        # now prepare data for training
        cfg = initialize_config_split(cfg)
        X, y = correct_data_label(X, y, cfg)
        Xs, ys = prepare_data_for_train(X, y, cfg)

        print("\nx_train Angiography cube shape: {}".format(Xs[0][0].shape))
        print("x_train Structure OCT cube shape: {}".format(Xs[0][1].shape))
        print("x_train B scan shape: {}".format(Xs[0][2].shape))
        print("y_train onehot shape: {}".format(ys[0].shape))

        print("\nx_valid Angiography cube shape: {}".format(Xs[1][0].shape))
        print("x_valid Structure OCT cube shape: {}".format(Xs[1][1].shape))
        print("x_valid B scan shape: {}".format(Xs[1][2].shape))
        print("y_valid onehot shape: {}".format(ys[1].shape))

        print("\nx_test Angiography cube shape: {}".format(Xs[2][0].shape))
        print("x_test Structure OCT cube shape: {}".format(Xs[2][1].shape))
        print("x_test B scan shape: {}".format(Xs[2][2].shape))
        print("y_test onehot shape: {}".format(ys[2].shape))

        # finally set the training parameters
        cfg = initialize_config_training(cfg, bool_debug=cfg.bool_debug)
        model = get_model(cfg.str_arch, cfg)
        callbacks = get_callbacks(cfg)

        h = model.fit(Xs[0], ys[0], batch_size=cfg.batch_size, epochs=cfg.n_epoch, verbose=2, callbacks=callbacks,
                      validation_data=(Xs[1], ys[1]), shuffle=False, validation_batch_size=Xs[1][0].shape[0])

        # Now perform prediction
        train_set_score = model.evaluate(Xs[0], ys[0], callbacks=callbacks, verbose=0)
        valid_set_score = model.evaluate(Xs[1], ys[1], callbacks=callbacks, verbose=0)
        test_set_score = model.evaluate(Xs[2], ys[2], callbacks=callbacks, verbose=0)

        vec_train_acc.append(train_set_score[1])
        vec_valid_acc.append(valid_set_score[1])
        vec_test_acc.append(test_set_score[1])

        if cfg.num_classes == 2:
            y_true = ys[-1]
            y_pred_logits = model.predict(Xs[2])

            y_pred = y_pred_logits.copy()
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
            y_pred = y_pred.reshape(-1)

            # now transform the logits into a matrix of two class probabilities and append
            y_pred_logits_both = np.concatenate([1 - y_pred_logits, y_pred_logits], axis=1)

        else:
            raise ValueError("The number of classes should be two")

        vec_y_true.append(y_true)
        vec_y_pred.append(y_pred)
        vec_y_pred_prob.append(y_pred_logits_both)
        mat_vec_idx_absolute_test.append(cfg.vec_idx_absolute[2])

        Xs = []
        ys = []

    print('\n' + '=' * 80)
    print("Average train set accuracy: {} + ".format(np.mean(vec_train_acc)), np.std(vec_train_acc))
    print("Average valid set accuracy: {} + ".format(np.mean(vec_valid_acc)), np.std(vec_valid_acc))
    print("Average test set accuracy: {} + ".format(np.mean(vec_test_acc)), np.std(vec_test_acc))

    # now concatenate the results together
    y_true = np.concatenate(vec_y_true, axis=0)
    y_pred = np.concatenate(vec_y_pred, axis=0)
    y_pred_prob = np.concatenate(vec_y_pred_prob, axis=0)
    vec_idx_absolute_test = np.concatenate(mat_vec_idx_absolute_test, axis=0)

    # sanity check
    if not y_true.shape[0] == y_pred.shape[0] == y_pred_prob.shape[0] == vec_idx_absolute_test.shape[0]:
        raise ValueError("These should be equal")

    # print the overall test set accuracy
    print("\nOverall test set accuracy: {}".format(np.sum(y_true == y_pred) / len(y_true)))

    # now load the preprocessed data and the label
    f_data_handle = "preproc_data_{}_{}.pkl".format(cfg_in.vec_idx_patient[0], cfg_in.vec_idx_patient[1])
    f_data = cfg_in.d_preproc / f_data_handle
    with open(str(f_data), 'rb') as handle:
        X = pickle.load(handle)

    y = generate_labels(cfg_in.str_feature, cfg_in, bool_append_csv_to_cfg=True)

    # now prepare data for training
    cfg_in = initialize_config_split(cfg_in)
    X, y = correct_data_label(X, y, cfg_in)
    _, _ = prepare_data_for_train(X, y, cfg_in)
    X, y = None, None

    # append the hyperparameters to the cfg structure
    cfg_in.vec_train_acc = vec_train_acc
    cfg_in.vec_valid_acc = vec_valid_acc
    cfg_in.vec_test_acc = vec_test_acc

    cfg_in.y_test_true = y_true
    cfg_in.y_test_pred = y_pred
    cfg_in.y_test_pred_prob = y_pred_prob
    cfg_in.vec_idx_absolute_test = vec_idx_absolute_test

    cfg_in.vec_y_true = vec_y_true
    cfg_in.vec_y_pred = vec_y_pred
    cfg_in.vec_y_pred_prob = vec_y_pred_prob
    cfg_in.mat_vec_idx_absolute_test = mat_vec_idx_absolute_test

    # now get the configuration right for saving the output
    cfg_in.str_model = cfg_in.str_arch
    cfg_in.f_model = "{}_{}".format(cfg_in.str_arch, time.strftime("%Y%m%d_%H%M%S"))
    cfg_in.p_figure = cfg_in.d_model / cfg_in.str_model / cfg_in.f_model
    cfg_in.p_figure.mkdir(parents=True, exist_ok=True)
    cfg_in.p_cfg = cfg_in.p_figure

    # plot the confusion matrices
    plot_raw_conf_matrix(y_true, y_pred, cfg_in, save=True, f_figure=cfg_in.str_arch)
    plot_norm_conf_matrix(y_true, y_pred, cfg_in, save=True, f_figure=cfg_in.str_arch)

    # save the cfg, which contains configurations and results
    save_cfg(cfg_in, overwrite=True)

    # save the mat file, which contains the binary results
    save_mat(cfg_in, overwrite=True)


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
    cfg_in.binary_mode = 0

    # whether or not to make the training set balanced - note this will give you imbalanced test set
    cfg_in.balanced = True

    # specify how many times to train the repeated models
    cfg_in.n_repeats = 1

    # now start the repeated training
    cfg_in.bool_debug = True
    cfg_in.str_arch = 'arch_010'
    run_repeated_training_binary(cfg_in)
