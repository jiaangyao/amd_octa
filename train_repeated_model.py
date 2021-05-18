import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from config.load_config import get_config
from preprocess import preprocess
from model import get_model, get_callbacks
from plotting import plot_raw_conf_matrix, plot_norm_conf_matrix
import time


# Configuring the files here for now
cfg = get_config(filename=pathlib.Path(os.getcwd()) / 'config' / 'default_config.yml')
cfg.d_data = pathlib.Path('/home/jyao/local/data/amd_octa/FinalData/')
cfg.d_model = pathlib.Path('/home/jyao/local/data/amd_octa/trained_models/')
# cfg.d_data = pathlib.Path('/home/kavi/Downloads/amd_octa_data/patient_id/')
# cfg.d_model = pathlib.Path('/home/kavi/Downloads/amd_octa_data/trained_models/')

# specify the loading mode: 'csv' vs 'folder'
# if csv, then loading based on a csv file
# if folder, then loading based on existing folder structure
cfg.load_mode = 'csv'
# cfg.load_mode = 'folder'
cfg.d_csv = pathlib.Path('/home/jyao/local/data/amd_octa')
# cfg.d_csv = pathlib.Path('/home/kavi/Downloads/amd_octa_data/')
cfg.f_csv = 'DiseaseLabelsThrough305.csv'

# name of particular feature that will be used
# note if want to test for disease label then have to specify this to be disease
# otherwise it has to match what's in the CSV file column header
cfg.vec_all_str_feature = ['disease', 'IRF/SRF', 'Scar', 'GA', 'CNV', 'PED']

cfg.str_feature = 'disease'
cfg.vec_str_labels = ['Normal', 'NNV AMD', 'NV AMD']

# cfg.str_feature = 'Scar'
# cfg.vec_str_labels = ['Not Present', 'Possible', 'Present']

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
cfg.vec_str_layer_bscan3d = ['1', '2', '3', '4', '5']
cfg.dict_layer_order = {'Deep': 0,
                        'Avascular': 1,
                        'ORCC': 2,
                        'Choriocapillaris': 3,
                        'Choroid': 4}
cfg.dict_layer_order_bscan3d = {'1': 0,
                                '2': 1,
                                '3': 2,
                                '4': 3,
                                '5': 4}
cfg.str_bscan_layer = 'Flow'
cfg.dict_str_patient_label = {}

cfg.downscale_size = [256, 256]
cfg.downscale_size_bscan = [450, 300]
cfg.crop_size = [int(np.round(cfg.downscale_size_bscan[0] * 1.5/7.32)),
                 int(np.round(cfg.downscale_size_bscan[0] * 1.8/7.32))]
# cfg.crop_size = None
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
cfg.cv_mode = False
cfg.oversample = False
cfg.oversample_method = 'smote'
cfg.decimate = False
cfg.random_seed = 68
cfg.use_random_seed = True
cfg.binary_class = False
cfg.n_repeats = 10

# vec_idx_healthy = [1, 250]
# vec_idx_dry_amd = [1, 250]
# vec_idx_cnv = [1, 250]
vec_idx_patient = [1, 310]

vec_train_acc = []
vec_valid_acc = []
vec_test_acc = []

vec_y_true = []
vec_y_pred = []
vec_model = []


np.random.seed(cfg.random_seed)


if cfg.str_feature != 'disease':
    raise Exception('You should run the feature training in CV mode')

for i in range(cfg.n_repeats):
    print("\n\nIteration: {}".format(i + 1))

    # Preprocessing
    #Xs, ys = preprocess(vec_idx_healthy, vec_idx_dry_amd, vec_idx_cnv, cfg)
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

    model = get_model('arch_022', cfg)
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
        y_pred = model.predict(Xs[2])
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = y_pred.reshape(-1)
    else:
        y_true = np.argmax(ys[-1], axis=1)
        y_pred = np.argmax(model.predict(Xs[2]), axis=1)

    vec_y_true.append(y_true)
    vec_y_pred.append(y_pred)

    Xs = []
    ys = []

print("Average train set accuracy: {} + ".format(np.mean(vec_train_acc)), np.std(vec_train_acc))
print("Average valid set accuracy: {} + ".format(np.mean(vec_valid_acc)), np.std(vec_valid_acc))
print("Average test set accuracy: {} + ".format(np.mean(vec_test_acc)), np.std(vec_test_acc))

y_true = np.concatenate(vec_y_true, axis=0)
y_pred = np.concatenate(vec_y_pred, axis=0)

f_model = "{}_{}".format(cfg.str_model, time.strftime("%Y%m%d_%H%M%S"))
# cfg.p_figure = pathlib.Path('/home/jyao/Downloads/conf_matrix_repeated_seed_fixed/') / cfg.str_model / f_model
cfg.p_figure = pathlib.Path('/home/kavi/Downloads/conf_matrix_repeated_seed_fixed/') / cfg.str_model / f_model
cfg.p_figure.mkdir(exist_ok=True, parents=True)

plot_raw_conf_matrix(y_true, y_pred, cfg, save=True, f_figure=cfg.str_model)
plot_norm_conf_matrix(y_true, y_pred, cfg, save=True, f_figure=cfg.str_model)
