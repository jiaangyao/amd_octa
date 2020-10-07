import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config.load_config import get_config
from preprocess import preprocess
from model import get_model, get_callbacks


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

cfg.downscale_size = [350, 350]
cfg.per_train = 0.8
cfg.per_valid = 0.1
cfg.per_test = 0.1

cfg.n_epoch = 100
cfg.batch_size = 1
cfg.es_patience = 3
cfg.es_min_delta = 1e-5
cfg.lr = 1e-3

vec_idx_healthy = [1, 150]
vec_idx_dry_amd = [1, 150]
vec_idx_cnv = [1, 150]

# Preprocessing
Xs, ys = preprocess(vec_idx_healthy, vec_idx_dry_amd, vec_idx_cnv, cfg)

# Get and train model
# from model import structure_conv3d, angiography_conv3d, bscan_conv2d
#
# t1 = structure_conv3d('arch_001', cfg)
# t2 = angiography_conv3d('arch_001', cfg)
# t3 = bscan_conv2d('arch_001', cfg)

model = get_model('arch_001', cfg)
callbacks = get_callbacks(cfg)

h = model.fit(Xs[0], ys[0], batch_size=cfg.batch_size, epochs=cfg.n_epoch, verbose=2, callbacks=callbacks,
              validation_data=(Xs[1], ys[1]), shuffle=True, validation_batch_size=Xs[1][0].shape[0])

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
score = model.evaluate(Xs[2], ys[2], callbacks=callbacks, verbose=0)
print("Test loss: {:2f}".format(score[0]))
print("Test acc: {}".format(score[1]))

y_test_pred = np.round(model.predict(Xs[2], callbacks=callbacks)).astype(np.int32).reshape(-1)
test_acc = np.mean(ys[2] == y_test_pred)
print("Test acc: {}".format(test_acc))

print('nothing')