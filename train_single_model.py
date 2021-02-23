import os
import pathlib
import numpy as np

from config.load_config import get_config
from preprocess import preprocess
from model import get_model, get_callbacks
from utils.io_funcs import *
from plotting import plot_training_loss, plot_training_acc, plot_raw_conf_matrix, plot_norm_conf_matrix


# Configuring the files here for now
cfg = get_config(filename=pathlib.Path(os.getcwd()) / 'config' / 'default_config.yml')
# cfg.d_data = pathlib.Path('/home/jyao/local/data/amd_octa/orig/')
cfg.d_data = pathlib.Path('/home/jyao/local/data/amd_octa/patient_id/')
cfg.d_model = pathlib.Path('/home/jyao/local/data/amd_octa/trained_models/')

# specify the loading mode: 'csv' vs 'folder'
# if csv, then loading based on a csv file
# if folder, then loading based on existing folder structure
cfg.load_mode = 'csv'
# cfg.load_mode = 'folder'
cfg.d_csv = pathlib.Path('/home/jyao/local/data/amd_octa/')
cfg.f_csv = 'BookMod.csv'

# name of particular feature that will be used
# note if want to test for disease label then have to specify this to be disease
# otherwise it has to match what's in the CSV file column header
# cfg.str_feature = 'disease'
cfg.str_feature = 'IRF/SRF'
cfg.vec_all_str_feature = ['disease', 'IRF/SRF', 'Scar', 'GA', 'CNV', 'PED']
cfg.vec_str_labels = ['Not Present', 'Possible', 'Present']
# cfg.vec_str_labels = ['Normal', 'NNV AMD', 'NV AMD']

cfg.str_healthy = 'Normal'
cfg.label_healthy = 0
cfg.str_dry_amd = 'Dry AMD'
cfg.label_dry_amd = 1
cfg.str_cnv = 'CNV'
cfg.label_cnv = 2
cfg.num_classes = 3

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
cfg.overwrite = True

cfg.balanced = False
cfg.oversample = False
cfg.oversample_method = 'smote'
cfg.random_seed = 68
cfg.use_random_seed = True
cfg.binary_class = False

vec_idx_patient = [1, 250]

# Preprocessing
Xs, ys = preprocess(vec_idx_patient, cfg)

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

# Get and train model
model = get_model('arch_009', cfg)
callbacks = get_callbacks(cfg)

h = model.fit(Xs[0], ys[0], batch_size=cfg.batch_size, epochs=cfg.n_epoch, verbose=2, callbacks=callbacks,
              validation_data=(Xs[1], ys[1]), shuffle=False, validation_batch_size=Xs[1][0].shape[0])
cfg.history = h.history

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

if cfg.num_classes == 2:
    y_true = ys[-1]
    y_pred = model.predict(Xs[2])
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
else:
    y_true = np.argmax(ys[-1], axis=1)
    y_pred = np.argmax(model.predict(Xs[2]), axis=1)

cfg.y_test_true = y_true
cfg.y_test_pred = y_pred

# plot the confusion matrices
plot_raw_conf_matrix(y_true, y_pred, cfg, save=True)
plot_norm_conf_matrix(y_true, y_pred, cfg, save=True)

# save the cfg, which contains configurations and results
save_cfg(cfg, overwrite=True)
