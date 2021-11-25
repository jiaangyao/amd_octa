import os
from pathlib import Path

from config.load_config import get_config
from preproc import preprocess
from modeling.model import get_model, get_callbacks
from save_load.io_funcs import *
import matplotlib.pyplot as plt
from analysis.plotting import plot_raw_conf_matrix

# Configuring the files here for now
cfg = get_config(filename=Path(os.getcwd()) / 'config' / 'default_config.yml')
cfg.d_data = Path('/home/jyao/Local/amd_octa/')
cfg.d_model = Path('/home/jyao/Local/amd_octa/trained_models/')

cfg.str_healthy = 'Normal'
cfg.label_healthy = 0
cfg.str_dry_amd = 'Dry AMD'
cfg.label_dry_amd = 1
cfg.str_cnv = 'CNV'
cfg.label_cnv = 2
cfg.num_classes = 3
cfg.vec_str_labels = ['Normal', 'NNV AMD', 'NV AMD']

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

cfg.balanced = True
cfg.oversample = False
cfg.oversample_method = 'smote'
cfg.random_seed = 68
cfg.use_random_seed = True
cfg.binary_class = False

vec_idx_healthy = [1, 250]
vec_idx_dry_amd = [1, 250]
vec_idx_cnv = [1, 250]

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

# Get model
model = get_model('arch_022', cfg)
callbacks = get_callbacks(cfg)

# Load model weights
# load_model(model, cfg, '20201025_114808')   # arch_009
# saved_cfg = load_config(cfg, '20201025_114808')

# load_model(model, cfg, '20201025_113418')   # arch_010
# saved_cfg = load_config(cfg, '20201025_113418')

load_model(model, cfg, '20201024_192227')   # arch_022
saved_cfg = load_config(cfg, '20201024_192227')

# now get the true and predicted labels for the test set
y_test_true = saved_cfg.y_test_true
y_test_pred = saved_cfg.y_test_pred

# extract the patient name
idx_test = 7
str_patient = saved_cfg.vec_str_patient[saved_cfg.vec_idx_absolute[2][idx_test]]
x_test_idx_test = Xs[2][1][idx_test, ...]

plt.figure()
plt.imshow(x_test_idx_test[:, :, 2, :])

# plot the raw confusion matrix for reference
plot_raw_conf_matrix(y_test_true, y_test_pred, cfg, save=False)

print('Done')
