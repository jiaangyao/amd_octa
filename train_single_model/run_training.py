import os
import pathlib
import pickle
import time

import numpy as np

from config.load_config import get_config
from config.config_utils import initialize_config_preproc, initialize_config_split, initialize_config_training
from preproc.preprocess import generate_labels, correct_data_label
from preproc.train_val_test_split import prepare_data_for_train
from modeling.model import get_model, get_callbacks
from save_load.io_funcs import save_model, save_cfg
from utils.get_patient_id import get_patient_id_by_label
from analysis.plotting import plot_training_loss, plot_training_acc, plot_raw_conf_matrix, plot_norm_conf_matrix


def run_training(Xs, ys, cfg):
    print("\nx_train Angiography cube shape: {}".format(Xs[0][0].shape))
    print("x_train Structure OCT cube shape: {}".format(Xs[0][1].shape))
    print("x_train B scan shape: {}".format(Xs[0][2].shape))
    print("x_train 3D B scan shape: {}".format(Xs[0][3].shape))
    print("y_train onehot shape: {}".format(ys[0].shape))

    print("\nx_valid Angiography cube shape: {}".format(Xs[1][0].shape))
    print("x_valid Structure OCT cube shape: {}".format(Xs[1][1].shape))
    print("x_valid B scan shape: {}".format(Xs[1][2].shape))
    print("x_valid 3D B scan shape: {}".format(Xs[1][3].shape))
    print("y_valid onehot shape: {}".format(ys[1].shape))

    print("\nx_test Angiography cube shape: {}".format(Xs[2][0].shape))
    print("x_test Structure OCT cube shape: {}".format(Xs[2][1].shape))
    print("x_test B scan shape: {}".format(Xs[2][2].shape))
    print("x_test 3D B scan shape: {}".format(Xs[2][3].shape))
    print("y_test onehot shape: {}".format(ys[2].shape))

    # Get and train model
    model = get_model(cfg.str_arch, cfg)
    callbacks = get_callbacks(cfg)

    tic = time.time()
    h = model.fit(Xs[0], ys[0], batch_size=cfg.batch_size, epochs=cfg.n_epoch, verbose=2, callbacks=callbacks,
                  validation_data=(Xs[1], ys[1]), shuffle=False, validation_batch_size=Xs[1][0].shape[0])
    toc = time.time()
    cfg.history = h.history
    cfg.time_elapsed_train = toc - tic

    # save trained models
    save_model(model, cfg, overwrite=True, save_format='tf')

    # plotting training history
    plot_training_loss(h, cfg, save=True)
    plot_training_acc(h, cfg, save=True)

    # Now perform prediction
    train_set_score = model.evaluate(Xs[0], ys[0], callbacks=callbacks, verbose=0)
    valid_set_score = model.evaluate(Xs[1], ys[1], callbacks=callbacks, verbose=0)
    test_set_score = model.evaluate(Xs[2], ys[2], callbacks=callbacks, verbose=0)

    print("\nTrain set accuracy: {}".format(train_set_score[1]))
    print("Valid set accuracy: {}".format(valid_set_score[1]))
    print("Test set accuracy: {}".format(test_set_score[1]))

    cfg.vec_acc = [train_set_score[1], valid_set_score[1], test_set_score[1]]

    tic = time.time()
    if cfg.num_classes == 2:
        y_true = ys[-1]
        y_pred_logits = model.predict(Xs[2])

        y_pred = y_pred_logits.copy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = y_pred.reshape(-1)

        # now transform the logits into a matrix of two class probabilities and append
        y_pred_logits_both = np.concatenate([1 - y_pred_logits, y_pred_logits], axis=1)
        cfg.y_test_pred_prob = y_pred_logits_both
    else:
        y_true = np.argmax(ys[-1], axis=1)
        y_pred_logits = model.predict(Xs[2])
        y_pred = np.argmax(y_pred_logits.copy(), axis=1)
        cfg.y_test_pred_prob = y_pred_logits
    toc = time.time()
    cfg.time_elapsed_test = toc - tic

    print("\nTraining time elapsed: {}".format(cfg.time_elapsed_train))
    print("Test time elapsed: {}".format(cfg.time_elapsed_test))

    # Printing out true and pred labels for log reg
    print('\nTest set: ground truth')

    print(y_true)

    print('Test set: prediction')

    print(y_pred)

    # now compute the test set accuracies
    test_acc = np.sum(y_true == y_pred) / len(y_true)
    cfg.test_acc_full = test_acc

    # Print out the patient IDs corresponding to the query
    # Here for example
    # if you are running 'disease' label and you set true_label_id = 0 and predicted_label_id = 2
    # then you would get the patients who are normal/healthy and but falsely classified as NV AMD
    # the true_label_id and predicted_label_id correspond to cfg.vec_str_labels defined above
    print(get_patient_id_by_label(y_true, y_pred, true_label_id=0, predicted_label_id=2, cfg=cfg))

    # you can also print multiple of these at the same time
    print(get_patient_id_by_label(y_true, y_pred, true_label_id=2, predicted_label_id=1, cfg=cfg))

    # Extra caveat: for feature labels since we don't have possible any more and since the classes
    # are automatically recasted to get the FN (patient has the feature but network predicts not present)
    # you need to do something like
    # print(get_patient_id_by_label(y_true, y_pred, true_label_id=1, predicted_label_id=0, cfg=cfg))

    cfg.y_test_true = y_true
    cfg.y_test_pred = y_pred

    # plot the confusion matrices
    plot_raw_conf_matrix(y_true, y_pred, cfg, save=True)
    plot_norm_conf_matrix(y_true, y_pred, cfg, save=True)

    # save the cfg, which contains configurations and results
    save_cfg(cfg, overwrite=True)


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
    cfg.str_feature = 'disease'

    # whether or not to make the training set balanced - note this will give you imbalanced test set
    cfg.balanced = True

    # specify model architecture and whether to use debug mode
    cfg.str_arch = 'arch_009'
    cfg.debug_mode = True

    # now load the preprocessed data and the label
    f_data_handle = "preproc_data_{}_{}.pkl".format(vec_idx_patient[0], vec_idx_patient[1])
    f_data = cfg_template.d_preproc / f_data_handle
    with open(str(f_data), 'rb') as handle:
        X = pickle.load(handle)

    y = generate_labels(cfg.str_feature, cfg, bool_append_csv_to_cfg=True)

    # now prepare data for training
    cfg = initialize_config_split(cfg)
    X, y = correct_data_label(X, y, cfg)
    Xs, ys = prepare_data_for_train(X, y, cfg)

    # finally set the training parameters
    cfg = initialize_config_training(cfg, bool_debug=cfg.debug_mode)
    cfg = run_training(Xs, ys, cfg)
