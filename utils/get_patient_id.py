import numpy as np


def get_patient_id_by_label(y_true, y_pred, true_label_id, predicted_label_id, cfg):
    vec_idx_true_label = (y_true == true_label_id)
    vec_idx_pred_label = (y_pred == predicted_label_id)
    vec_idx_intersect = np.logical_and(vec_idx_true_label, vec_idx_pred_label)

    if not cfg.cv_mode:
        vec_idx_absolute_test = cfg.vec_idx_absolute[-1]
    else:
        vec_idx_absolute_test = np.arange(0, len(y_true), 1)

    vec_idx_absolute_test_intersect = vec_idx_absolute_test[vec_idx_intersect]
    str_output = "\n\nTrue Label: {}, Predicted Label: {}\n".format(cfg.vec_str_labels[true_label_id],
                                                                    cfg.vec_str_labels[predicted_label_id])
    if not len(vec_idx_absolute_test_intersect) == 0:
        for idx_curr in vec_idx_absolute_test_intersect:
            str_output += cfg.vec_str_patients[idx_curr] + '\n'
    else:
        str_output += 'None\n'
    str_output += '\n'

    cfg.dict_str_patient_label['true_label={}, predicted_label={}'.format(true_label_id, predicted_label_id)] = str_output

    return str_output
