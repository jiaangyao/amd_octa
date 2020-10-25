import pathlib
import tensorflow
import time
import pickle


def save_model(model, cfg, overwrite=True, save_format='tf'):
    """
    save the trained model

    :param model: trained model
    :param cfg: configuration set by user
    :param bool overwrite: whether or not to overwrite any existing model weights
    :param str save_format: format to save the weights in
    """

    f_model = "{}_{}".format(cfg.str_model, time.strftime("%Y%m%d_%H%M%S"))
    p_model = cfg.d_model / cfg.str_model / f_model
    p_model.mkdir(parents=True, exist_ok=True)

    cfg.p_figure = p_model
    cfg.p_cfg = p_model

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
