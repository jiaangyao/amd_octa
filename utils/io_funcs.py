import pathlib
import tensorflow
import numpy as np
import time
import pickle


def save_model(model, cfg, overwrite=True, save_format='tf', idx_cv_fold=None):
    """
    save the trained model

    :param model: trained model
    :param cfg: configuration set by user
    :param bool overwrite: whether or not to overwrite any existing model weights
    :param str save_format: format to save the weights in
    :param int idx_cv_fold: only relevant for cross validation mode, specify which fold it is
    """
    if not cfg.cv_mode:
        f_model = "{}_{}".format(cfg.str_model, time.strftime("%Y%m%d_%H%M%S"))
        p_model = cfg.d_model / cfg.str_model / f_model
        p_model.mkdir(parents=True, exist_ok=True)

        cfg.p_cfg = p_model

    else:
        if idx_cv_fold == 0:
            f_model = "{}_{}".format(cfg.str_model, time.strftime("%Y%m%d_%H%M%S"))
            cfg.f_model_cv = f_model
        else:
            f_model = cfg.f_model_cv

        p_model = cfg.d_model / cfg.str_model / f_model / "fold_{}".format(idx_cv_fold)
        p_model.mkdir(parents=True, exist_ok=True)

        cfg.p_cfg = cfg.d_model / cfg.str_model / f_model
        cfg.p_figure_all = cfg.p_cfg

        cfg.p_out_csv = cfg.p_figure_all

    cfg.p_figure = p_model

    model.save_weights(filepath=str(p_model / f_model), overwrite=overwrite, save_format=save_format)


def save_cfg(cfg, overwrite=True):
    pf_cfg = cfg.p_cfg / 'cfg_file'
    if not (pf_cfg.exists() and not overwrite):
        with open(str(cfg.p_cfg / 'cfg_file'), 'wb') as handle:
            pickle.dump(cfg, handle)


def load_model(model, cfg, name, **kwargs):
    """
    load model weights from a saved weight file

    :param model: trained model
    :param cfg: configuration set by user
    :param name: name of the model after the underscore
    :param kwargs: additional arguments accepted by the tensorflow.keras.Model.load_weights() function
    """

    f_model = "{}_{}".format(cfg.str_model, name)
    p_model = cfg.d_model / cfg.str_model / f_model
    cfg.p_figure = p_model
    if not p_model.exists():
        raise Exception('No saved models are available: check path setting')

    model.load_weights(filepath=str(p_model / f_model), **kwargs)


def load_config(cfg, name):
    """
    load saved configuration dictionaries

    :param cfg: configuration set by user
    :param name: name of the model after the underscore
    """

    f_model = "{}_{}".format(cfg.str_model, name)
    p_model = cfg.d_model / cfg.str_model / f_model
    cfg.p_figure = p_model
    if not p_model.exists():
        raise Exception('No saved configs are available: check path setting')

    with open(str(p_model / 'cfg_file'), 'rb') as handle:
        saved_cfg = pickle.load(handle)

    return saved_cfg


def save_csv(y_true_all, y_pred_all, cfg):
    """
    Save the prediction as a csv file

    :param y_true_all: Ground truth label parsed from csv file
    :param y_pred_all: Predicted label from cross validation mode
    :param cfg:
    :return:
    """
    pd_csv = cfg.pd_csv.copy()
    out_csv = cfg.out_csv.copy()
    if cfg.str_feature == 'disease':
        cfg.y_unique_label = cfg.y_unique_label + 1
    for i in range(len(cfg.vec_out_csv_idx)):
        y_true_all_curr = y_true_all[i]
        y_pred_all_curr = y_pred_all[i]
        out_csv.loc[cfg.vec_out_csv_idx[i][0], out_csv.columns[cfg.vec_out_csv_idx[i][1]]] = cfg.y_unique_label[
            int(y_pred_all_curr)]
        if not pd_csv.loc[cfg.vec_out_csv_idx[i][0], pd_csv.columns[cfg.vec_out_csv_idx[i][1]]] == cfg.y_unique_label[
            int(y_true_all_curr)]:
            raise Exception("Ground truth label not equal to csv file")

    idx_col_OD_feature = cfg.vec_csv_col[-2]
    idx_col_OS_feature = cfg.vec_csv_col[-1]

    vec_idx_row_all = np.arange(out_csv.shape[0])
    idx_valid_OD_out_csv = vec_idx_row_all[np.logical_not(np.isnan(out_csv.loc[:, out_csv.columns[idx_col_OD_feature]]))]
    idx_valid_OS_out_csv = vec_idx_row_all[np.logical_not(np.isnan(out_csv.loc[:, out_csv.columns[idx_col_OS_feature]]))]

    if cfg.str_feature == 'disease':
        idx_col_OD_valid = cfg.vec_csv_col[1]
        idx_col_OS_valid = cfg.vec_csv_col[2]

        idx_invalid_OD_out_csv = vec_idx_row_all[pd_csv.loc[:, pd_csv.columns[idx_col_OD_valid]].to_numpy() == 0]
        idx_invalid_OS_out_csv = vec_idx_row_all[pd_csv.loc[:, pd_csv.columns[idx_col_OS_valid]].to_numpy() == 0]

        out_csv.loc[idx_invalid_OD_out_csv, pd_csv.columns[idx_col_OD_valid]] = 0
        out_csv.loc[idx_invalid_OS_out_csv, pd_csv.columns[idx_col_OS_valid]] = 0

    idx_valid_OD_pd_csv = vec_idx_row_all[np.logical_not(np.isnan(pd_csv.loc[:, out_csv.columns[idx_col_OD_feature]]))]
    idx_valid_OS_pd_csv = vec_idx_row_all[np.logical_not(np.isnan(pd_csv.loc[:, out_csv.columns[idx_col_OS_feature]]))]

    if not (np.allclose(idx_valid_OD_pd_csv, idx_valid_OD_out_csv) and np.allclose(idx_valid_OS_pd_csv,
                                                                                   idx_valid_OS_out_csv)):
        raise Exception('Error editing the output csv file')

    cfg.f_out_csv = 'predLabel_{}.csv'.format(cfg.f_model_cv)
    cfg.p_out_csv = cfg.p_figure_all
    cfg.out_csv = out_csv
    out_csv.to_csv(cfg.p_out_csv / cfg.f_out_csv)
