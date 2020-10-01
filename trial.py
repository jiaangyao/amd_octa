import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config.load_config import get_config
from preprocess import preprocess
from model import get_model, get_callbacks


# Configuring the files here for now
cfg = get_config(filename=Path(os.getcwd()) / 'config' / 'default_config.yml')
cfg.d_data = Path('/home/jyao/Local/amd_octa/raw_data/')
cfg.d_model = Path('/home/jyao/Local/amd_octa/trained_models/')

cfg.str_healthy = 'normalPatient'
cfg.label_healthy = 0
cfg.str_dry_amd = 'dryAMDPatient'
cfg.label_dry_amd = 1

cfg.downscale_size = [350, 350]
cfg.per_train = 0.6
cfg.per_valid = 0.2
cfg.per_test = 0.2

cfg.n_epoch = 100
cfg.batch_size = 1
cfg.es_patience = 5
cfg.es_min_delta = 1e-5
cfg.lr = 1e-5

vec_idx_run_healthy = np.append(np.arange(1, 8, 1), 10)
vec_idx_run_dry_amd = np.arange(24, 32, 1)

# Preprocessing
Xs, ys = preprocess(vec_idx_run_healthy, vec_idx_run_dry_amd, cfg)

# Get and train model
model = get_model('arch_001', cfg)
callbacks = get_callbacks(cfg)

h = model.fit(Xs[0], ys[0], batch_size=cfg.batch_size, epochs=cfg.n_epoch, verbose=1, callbacks=callbacks,
              validation_data=(Xs[1], ys[1]), shuffle=True, validation_batch_size=Xs[1].shape[0])

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
y_test_pred = np.round(model.predict(Xs[2], callbacks=callbacks)).astype(np.int32).reshape(-1)
test_acc = np.mean(ys[2] == y_test_pred)
print("Test acc: {}".format(test_acc))