from numba import jit, float64
import numpy as np
from math import log, exp, expm1, log1p

@jit(float64(float64[:], float64[:]), nopython=True, nogil=True)
def calc_prob(w, r):
    exp_w = np.exp(w)
    w_times_R = exp_w * r
    exp_p = w_times_R.sum()
    p = log(exp_p)
    return p


@jit(float64(float64), nopython=True, nogil=True)
def log1mexp_numba(a):
    if a > 0:
        print 'LOGEXP'
        return

    if a < -log(2.0):
        return log1p(-exp(a))
    else:
        return log(-expm1(a))


# mostly from:  http://nbviewer.ipython.org/gist/sebastien-bratieres/285184b4a808dfea7070

# # numba logging bug http://stackoverflow.com/questions/19112584/huge-errors-trying-numba
# import logging
# numba.codegen.debug.logger.setLevel(logging.INFO)

@jit(float64(float64[:]), nopython=True, nogil=True)
def lse_numba(a):
    result = 0.0
    largest_in_a = a.max()
    for i in range(a.shape[0]):
        result += exp(a[i] - largest_in_a)
    return log(result) + largest_in_a