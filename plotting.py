from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def plot_training_loss(h, cfg, save=True):
    plt.figure()
    plt.plot(np.arange(1, len(h.history['loss']) + 1, 1), h.history['loss'])
    plt.plot(np.arange(1, len(h.history['val_loss']) + 1, 1), h.history['val_loss'])
    plt.title("Loss history in training")
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')

    fig = plt.gcf()
    if save:
        f_figure = 'training_loss.eps'
        fig.savefig(str(cfg.p_figure / f_figure), format='eps')

        plt.close(fig)
    else:
        plt.show()


def plot_training_acc(h, cfg, save=True):
    plt.figure()
    plt.plot(np.arange(1, len(h.history['accuracy']) + 1, 1), h.history['accuracy'])
    plt.plot(np.arange(1, len(h.history['val_accuracy']) + 1, 1), h.history['val_accuracy'])
    plt.title("Accuracy history in training")
    plt.xlabel('Training Iterations')
    plt.ylabel('Accuracy')

    fig = plt.gcf()
    if save:
        f_figure = 'training_acc.eps'
        fig.savefig(str(cfg.p_figure / f_figure), format='eps')

        plt.close(fig)
    else:
        plt.show()


def plot_norm_conf_matrix(y_true, y_pred, cfg, save=True, f_figure=None):
    conf_matrix = confusion_matrix(y_true, y_pred)

    # plot the confusion matrix
    # normalize the matrix first
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    params = {
        'font.size': 15,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'text.usetex': False,
    }

    # params = {
    #     'font.size': 16,
    #     'axes.titlesize': 16,
    #     'axes.labelsize': 16,
    #     'xtick.labelsize': 16,
    #     'ytick.labelsize': 16,
    #     'text.usetex': False,
    # }
    rcParams.update(params)

    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix_norm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    if cfg.str_model == 'arch_009':
        str_title = 'Normalized Confusion Matrix: {}'.format('OCTA')
    elif cfg.str_model == 'arch_010':
        str_title = 'Normalized Confusion Matrix: {}'.format('OCTA+OCT')
    elif cfg.str_model == 'arch_012' or cfg.str_model == 'arch_013' or cfg.str_model == 'arch_023':
        str_title = 'Normalized Confusion Matrix: {}'.format('B-scan')
    elif cfg.str_model == 'arch_022':
        str_title = 'Normalized Confusion Matrix: {}'.format('OCTA+OCT+B-scan')
    else:
        str_title = 'Normalized Confusion Matrix'

    plt.title(str_title)

    thresh = np.max(conf_matrix_norm) / 1.5
    for i, j in itertools.product(range(conf_matrix_norm.shape[0]),
                                  range(conf_matrix_norm.shape[1])):
        plt.text(j, i, "{:0.3f}".format(conf_matrix_norm[i, j]),
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

    fig = plt.gcf()
    if save:
        if f_figure is not None:
            f_figure = 'norm_conf_matrix_{}.eps'.format(f_figure)
        else:
            f_figure = 'norm_conf_matrix.eps'
        fig.savefig(str(cfg.p_figure / f_figure), format='eps')
        plt.close(fig)
    else:
        plt.show()


def plot_raw_conf_matrix(y_true, y_pred, cfg, save=True, f_figure=None):
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_copy = np.array(conf_matrix, copy=True)

    # plot the confusion matrix
    # normalize the matrix first
    conf_matrix_norm = conf_matrix_copy.astype('float') / conf_matrix_copy.sum(axis=1)[:, np.newaxis]

    params = {
        'font.size': 15,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'text.usetex': False,
    }

    # params = {
    #     'font.size': 16,
    #     'axes.titlesize': 16,
    #     'axes.labelsize': 16,
    #     'xtick.labelsize': 16,
    #     'ytick.labelsize': 16,
    #     'text.usetex': False,
    # }
    rcParams.update(params)

    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix_norm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    if cfg.str_model == 'arch_009':
        str_title = 'Raw Confusion Matrix: {}'.format('OCTA')
    elif cfg.str_model == 'arch_010':
        str_title = 'Raw Confusion Matrix: {}'.format('OCTA+OCT')
    elif cfg.str_model == 'arch_012' or cfg.str_model == 'arch_013' or cfg.str_model == 'arch_023':
        str_title = 'Raw Confusion Matrix: {}'.format('B-scan')
    elif cfg.str_model == 'arch_022':
        str_title = 'Raw Confusion Matrix: {}'.format('OCTA+OCT+B-scan')
    else:
        str_title = 'Raw Confusion Matrix'

    plt.title(str_title)

    thresh = np.max(conf_matrix_norm) / 1.5
    for i, j in itertools.product(range(conf_matrix.shape[0]),
                                  range(conf_matrix.shape[1])):
        plt.text(j, i, "{}".format(conf_matrix[i, j]),
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

    fig = plt.gcf()
    if save:
        if f_figure is not None:
            f_figure = 'raw_conf_matrix_{}.eps'.format(f_figure)
        else:
            f_figure = 'raw_conf_matrix.eps'
        fig.savefig(str(cfg.p_figure / f_figure), format='eps')
        plt.close(fig)

    else:
        plt.show()

