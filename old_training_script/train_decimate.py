import os
from pathlib import Path

from config.load_config import get_config
from preproc import preprocess
from modeling.model import get_callbacks, get_model_binary
from analysis.plotting import plot_norm_conf_matrix, plot_raw_conf_matrix, plot_training_loss, plot_training_acc


# Configuring the files here for now
cfg = get_config(filename=Path(os.getcwd()) / 'config' / 'default_config.yml')
cfg.d_data = Path('/home/jyao/local/data/orig/amd_octa/')
cfg.d_model = Path('/home/jyao/local/data/orig/amd_octa/trained_models/')

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
cfg.overwrite = True

cfg.balanced = False
cfg.oversample = False
cfg.oversample_method = 'smote'
cfg.random_seed = 68
cfg.use_random_seed = True
cfg.binary_class = True
cfg.decimate = True

vec_idx_healthy = [1, 250]
vec_idx_dry_amd = [1, 250]
vec_idx_cnv = [1, 250]

if not cfg.decimate:
    raise Exception('Incorrect flag for decimation')

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

# decimate
n_train = Xs[0][0].shape[0]
n_train_decimate = round(n_train / 2)
x_train = Xs[0]
y_train = ys[0]

x_train[0] = x_train[0][:n_train_decimate, :, :, :, :]
x_train[1] = x_train[1][:n_train_decimate, :, :, :, :]
x_train[2] = x_train[2][:n_train_decimate, :, :, :]

y_train = y_train[:n_train_decimate, :]

# Get and train model
model = get_model_binary('arch_009_binary', cfg)
callbacks = get_callbacks(cfg)

h = model.fit(x_train, y_train, batch_size=cfg.batch_size, epochs=cfg.n_epoch, verbose=2, callbacks=callbacks,
              validation_data=(Xs[1], ys[1]), shuffle=False, validation_batch_size=Xs[1][0].shape[0])


plot_training_loss(h)
plot_training_acc(h)

# Now perform prediction
train_set_score = model.evaluate(x_train, y_train, callbacks=callbacks, verbose=0)
valid_set_score = model.evaluate(Xs[1], ys[1], callbacks=callbacks, verbose=0)
test_set_score = model.evaluate(Xs[2], ys[2], callbacks=callbacks, verbose=0)

print("\nAverage train set accuracy: {}".format(train_set_score[1]))
print("Average valid set accuracy: {}".format(valid_set_score[1]))
print("Average test set accuracy: {}".format(test_set_score[1]))

y_true = ys[-1]
y_pred = model.predict(Xs[2])
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0

plot_raw_conf_matrix(y_true, y_pred, cfg)
plot_norm_conf_matrix(y_true, y_pred, cfg)