from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(h):
    plt.figure()
    plt.plot(np.arange(1, len(h.history['loss']) + 1, 1), h.history['loss'])
    plt.plot(np.arange(1, len(h.history['val_loss']) + 1, 1), h.history['val_loss'])
    plt.title("Loss history in training")
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.show()


def plot_training_acc(h):
    plt.figure()
    plt.plot(np.arange(1, len(h.history['accuracy']) + 1, 1), h.history['accuracy'])
    plt.plot(np.arange(1, len(h.history['val_accuracy']) + 1, 1), h.history['val_accuracy'])
    plt.title("Accuracy history in training")
    plt.xlabel('Training Iterations')
    plt.ylabel('Accuracy')
    plt.show()


def plot_norm_conf_matrix(y_true, y_pred, cfg):
    conf_matrix = confusion_matrix(y_true, y_pred)

    # plot the confusion matrix
    # normalize the matrix first
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    cmap = plt.get_cmap('Blues')
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix_norm, interpolation='nearest', cmap=cmap)
    plt.title('Normalized Confusion Matrix')
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

    plt.show()


def plot_raw_conf_matrix(y_true, y_pred, cfg):
    conf_matrix = confusion_matrix(y_true, y_pred)

    cmap = plt.get_cmap('Blues')
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title('Raw Confusion Matrix')
    plt.colorbar()

    thresh = np.max(conf_matrix) / 1.5
    for i, j in itertools.product(range(conf_matrix.shape[0]),
                                  range(conf_matrix.shape[1])):
        plt.text(j, i, "{}".format(conf_matrix[i, j]),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()

    ax = plt.gca()
    ax.set(xticks=np.arange(len(cfg.vec_str_labels)),
           yticks=np.arange(len(cfg.vec_str_labels)),
           xticklabels=cfg.vec_str_labels,
           yticklabels=cfg.vec_str_labels,
           ylabel="True label",
           xlabel="Predicted label")

    plt.show()