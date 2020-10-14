import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config.load_config import get_config
from preprocess import preprocess
from model import get_model, get_callbacks
from sklearn.metrics import confusion_matrix
import itertools


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
cfg.vec_str_labels = ['Normal', 'Dry Amd', 'CNV']

cfg.num_octa = 5
cfg.str_angiography = 'Angiography'
cfg.str_structural = 'Structure'
cfg.str_bscan = 'B-Scan'

cfg.vec_str_layer = ['Deep', 'Avascular', 'ORCC', 'Choriocapillaris', 'Choroid']
cfg.dict_layer_order = {'Deep': 0,
                        'Avascular': 1,
                        'ORCC': 2,
                        'Choriocapillaris': 3,
                        'Choroid': 4}
cfg.str_bscan_layer = 'Flow'

cfg.downscale_size = [512, 512]
cfg.per_train = 0.6
cfg.per_valid = 0.2
cfg.per_test = 0.2

cfg.n_epoch = 1000
cfg.batch_size = 8
cfg.es_patience = 20
cfg.es_min_delta = 1e-5
cfg.lr = 5e-5
cfg.lam = 1e-5

vec_idx_healthy = [1, 150]
vec_idx_dry_amd = [1, 150]
vec_idx_cnv = [1, 150]

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

# Get and train model
# from model import structure_conv3d, angiography_conv3d, bscan_conv2d
#
# t1 = structure_conv3d('arch_001', cfg)
# t2 = angiography_conv3d('arch_001', cfg)
# t3 = bscan_conv2d('arch_001', cfg)

model = get_model('arch_007', cfg)
callbacks = get_callbacks(cfg)

h = model.fit(Xs[0], ys[0], batch_size=cfg.batch_size, epochs=cfg.n_epoch, verbose=2, callbacks=callbacks,
              validation_data=(Xs[1], ys[1]), shuffle=False, validation_batch_size=Xs[1][0].shape[0])

plt.figure()
plt.plot(np.arange(1, len(h.history['loss']) + 1, 1), h.history['loss'])
plt.plot(np.arange(1, len(h.history['val_loss']) + 1, 1), h.history['val_loss'])
plt.title("Loss history in training")
plt.xlabel('Training Iterations')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(np.arange(1, len(h.history['accuracy']) + 1, 1), h.history['accuracy'])
plt.plot(np.arange(1, len(h.history['val_accuracy']) + 1, 1), h.history['val_accuracy'])
plt.title("Accuracy history in training")
plt.xlabel('Training Iterations')
plt.ylabel('Accuracy')
plt.show()

# Now perform prediction
train_set_score = model.evaluate(Xs[0], ys[0], callbacks=callbacks, verbose=0)
valid_set_score = model.evaluate(Xs[1], ys[1], callbacks=callbacks, verbose=0)
test_set_score = model.evaluate(Xs[2], ys[2], callbacks=callbacks, verbose=0)

print("Average train set accuracy: {}".format(train_set_score[1]))
print("Average valid set accuracy: {}".format(valid_set_score[1]))
print("Average test set accuracy: {}".format(test_set_score[1]))

y_true = np.argmax(ys[-1], axis=1)
y_pred = np.argmax(model.predict(Xs[2]), axis=1)
conf_matrix = confusion_matrix(y_true, y_pred)

# plot the confusion matrix
# normalize the matrix first
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# plot the matrix
cmap = plt.get_cmap('Blues')
fig = plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_norm, interpolation='nearest', cmap=cmap)
plt.title('Normalized Confusion Matrix cnn_ensemble')
plt.colorbar()

thresh = np.max(conf_matrix_norm) / 1.5
for i, j in itertools.product(range(conf_matrix_norm.shape[0]),
                              range(conf_matrix_norm.shape[1])):

    plt.text(j, i, "{:0.4f}".format(conf_matrix_norm[i, j]),
             horizontalalignment="center",
             color="white" if conf_matrix_norm[i, j] > thresh else "black")

plt.tight_layout()

ax = plt.gca()
ax.set(xticks=np.arange(len(cfg.vec_str_labels)),
       yticks=np.arange(len(cfg.vec_str_labels)),
       xticklabels=cfg.vec_str_labels,
       yticklabels=cfg.vec_str_labels,
       ylabel="True label",
       xlabel="Predicted label")


## Confusion matrix
## and model visualization
print('nothing')