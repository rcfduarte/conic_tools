import numpy as np
from tqdm import tqdm
from sklearn import metrics as met

from .timeseries import autocorrelation_function


def pairwise_dist(matrix):
    return np.tril(met.pairwise_distances(matrix))


def cross_trial_cc(response_matrix, display=True):
    if display:
        print("Computing autocorrelations..")
    units = response_matrix.shape[0]

    r = []
    for nn in tqdm(range(units)):
        rr = autocorrelation_function(response_matrix[nn, :])
        if not np.isnan(np.mean(rr)):
            r.append(rr)  # [1:])
    return np.array(r)
