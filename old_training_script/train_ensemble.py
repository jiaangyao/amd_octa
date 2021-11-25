import os
from pathlib import Path
import numpy as np

from config.load_config import get_config
from preproc import preprocess
from modeling.model import get_model, get_callbacks
from analysis.plotting import plot_norm_conf_matrix, plot_raw_conf_matrix
from scipy.stats import mode


# Configuring the files here for now
cfg = get_config(filename=Path(os.getcwd()) / 'config' / 'default_config.yml')
cfg.d_data = Path('/home/jyao/local/data/orig/amd_octa/')
cfg.d_model = Path('/home/jyao/local/data/orig/amd_octa/trained_models/')

cfg.str_healthy = 'Normal'
cfg.label_healthy = 0
cfg.str_dry_amd = 'Dry AMD'
cfg.label_dry_amd = 1
cfg.str_cnv = 'CNV'
cfg.label_cnv = 2
cfg.num_classes = 3
cfg.vec_str_labels = ['Normal', 'Dry Amd', 'CNV']

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
cfg.oversample = False
cfg.oversample_method = 'smote'
cfg.random_seed = 68
cfg.use_random_seed = True
cfg.binary_class = False

cfg.n_ensemble = 5

vec_idx_healthy = [1, 150]
vec_idx_dry_amd = [1, 150]
vec_idx_cnv = [1, 150]

vec_train_acc = []
vec_valid_acc = []
vec_test_acc = []

vec_y_true = []
vec_y_pred = []
vec_model = []

# Preprocessing
Xs, ys = preprocess(vec_idx_healthy, vec_idx_dry_amd, vec_idx_cnv, cfg)

for i in range(cfg.n_ensemble):

    print("\n\nIteration: {}".format(i + 1))
    model = get_model('arch_011', cfg)
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

    y_true = np.argmax(ys[-1], axis=1)
    y_pred = np.argmax(model.predict(Xs[2]), axis=1)

    vec_y_true.append(y_true)
    vec_y_pred.append(y_pred)

    vec_model.append(model)

print("Average train set accuracy: {} + ".format(np.mean(vec_train_acc)), np.std(vec_train_acc))
print("Average valid set accuracy: {} + ".format(np.mean(vec_valid_acc)), np.std(vec_valid_acc))
print("Average test set accuracy: {} + ".format(np.mean(vec_test_acc)), np.std(vec_test_acc))

y_true = np.concatenate(vec_y_true, axis=0)
y_pred = np.concatenate(vec_y_pred, axis=0)

plot_raw_conf_matrix(y_true, y_pred, cfg)
plot_norm_conf_matrix(y_true, y_pred, cfg)

mat_pred = []

for i in range(len(vec_model)):
    curr_model = vec_model[i]
    y_pred_curr = np.argmax(curr_model.predict(Xs[2]), axis=1)

    mat_pred.append(y_pred_curr)
mat_pred = np.stack(mat_pred, axis=0)

y_pred_mode = mode(mat_pred, axis=0).mode.reshape(-1)
y_true_alt = np.argmax(ys[-1], axis=1)
ensemble_acc = np.sum(y_pred_mode == y_true_alt) / len(y_true_alt)

plot_raw_conf_matrix(y_true_alt, y_pred_mode, cfg)
plot_norm_conf_matrix(y_true_alt, y_pred_mode, cfg)

print('nothing')