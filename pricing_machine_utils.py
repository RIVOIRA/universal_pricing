import numpy as np


def generate_row_interval(val_min, val_max, row_length):
    interval = np.linspace(val_min, val_max, row_length)
    nb_rows = interval.shape[0]
    np.reshape(interval, (nb_rows, 1))
    return interval
