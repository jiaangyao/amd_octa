import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config.load_config import get_config
from preprocess import preprocess
from model import get_model, get_callbacks, get_model_binary
from plotting import plot_norm_conf_matrix, plot_raw_conf_matrix

# Configuring the files here for now
cfg = get_config(filename=Path(os.getcwd()) / 'config' / 'default_config.yml')
cfg.d_data = Path('/home/jyao/Local/amd_octa/')
cfg.d_model = Path('/home/jyao/Local/amd_octa/trained_models/')

cfg.str_dry_amd = 'Dry AMD'
cfg.label_dry_amd = 0
cfg.str_cnv = 'CNV'
cfg.label_cnv = 1
cfg.num_classes = 2
cfg.vec_str_labels = ['Dry Amd', 'CNV']

cfg.num_octa = 5
cfg.str_angiography = 'Angiography'
cfg.str_structure = 'Structure'
cfg.str_bscan = 'B-Scan'

cfg.vec_str_layer = ['Deep', 'Avascular', 'ORCC', 'Choriocapillaris', 'Choroid']
cfg.dict_layer_order = {'Deep': 0,
                        'Avascular': 1,
                        'ORCC': 2,
                        'Choriocapillaris': 3,
                        'Choroid': 4}
cfg.str_bscan_layer = 'Flow'

cfg.downscale_size = [256, 256]
cfg.per_train = 0.6
cfg.per_valid = 0.2
cfg.per_test = 0.2

cfg.n_epoch = 1000
cfg.batch_size = 8
cfg.es_patience = 20
cfg.es_min_delta = 1e-5
cfg.lr = 5e-5
cfg.lam = 1e-5
cfg.balanced = False
cfg.oversample = False
cfg.oversample_method = 'smote'
cfg.decimate = False
cfg.random_seed = 68
cfg.use_random_seed = False
cfg.binary_class = True

cfg.n_repeats = 5

vec_idx_healthy = [1, 250]
vec_idx_dry_amd = [1, 250]
vec_idx_cnv = [1, 250]

vec_train_acc = []
vec_valid_acc = []
vec_test_acc = []

vec_y_true = []
vec_y_pred = []
vec_model = []


for i in range(cfg.n_repeats):

    print("\n\nIteration: {}".format(i + 1))
    # Preprocessing
    Xs, ys = preprocess(vec_idx_healthy, vec_idx_dry_amd, vec_idx_cnv, cfg)

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

    n_train = Xs[0][0].shape[0]
    #
    if cfg.decimate:
        n_train_decimate = round(n_train / 2)
        x_train = Xs[0]
        y_train = ys[0]

        x_train[0] = x_train[0][:n_train_decimate, :, :, :, :]
        x_train[1] = x_train[1][:n_train_decimate, :, :, :, :]
        x_train[2] = x_train[2][:n_train_decimate, :, :, :]

        y_train = y_train[:n_train_decimate, :]

    model = get_model_binary('arch_009_binary', cfg)
    callbacks = get_callbacks(cfg)

    if not cfg.decimate:
        h = model.fit(Xs[0], ys[0], batch_size=cfg.batch_size, epochs=cfg.n_epoch, verbose=2, callbacks=callbacks,
                      validation_data=(Xs[1], ys[1]), shuffle=False, validation_batch_size=Xs[1][0].shape[0])

    else:
        h = model.fit(x_train, y_train, batch_size=cfg.batch_size, epochs=cfg.n_epoch, verbose=2, callbacks=callbacks,
                      validation_data=(Xs[1], ys[1]), shuffle=False, validation_batch_size=Xs[1][0].shape[0])

    # Now perform prediction
    if not cfg.decimate:
        train_set_score = model.evaluate(Xs[0], ys[0], callbacks=callbacks, verbose=0)
    else:
        train_set_score = model.evaluate(x_train, y_train, callbacks=callbacks, verbose=0)
    valid_set_score = model.evaluate(Xs[1], ys[1], callbacks=callbacks, verbose=0)
    test_set_score = model.evaluate(Xs[2], ys[2], callbacks=callbacks, verbose=0)

    vec_train_acc.append(train_set_score[1])
    vec_valid_acc.append(valid_set_score[1])
    vec_test_acc.append(test_set_score[1])

    y_true = ys[-1]
    y_pred = model.predict(Xs[2])
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    vec_y_true.append(y_true)
    vec_y_pred.append(y_pred)

    Xs = []
    ys = []

print("Average train set accuracy: {} + ".format(np.mean(vec_train_acc)), np.std(vec_train_acc))
print("Average valid set accuracy: {} + ".format(np.mean(vec_valid_acc)), np.std(vec_valid_acc))
print("Average test set accuracy: {} + ".format(np.mean(vec_test_acc)), np.std(vec_test_acc))

y_true = np.concatenate(vec_y_true, axis=0)
y_pred = np.concatenate(vec_y_pred, axis=0)

plot_raw_conf_matrix(y_true, y_pred, cfg)
plot_norm_conf_matrix(y_true, y_pred, cfg)

print('nothing')