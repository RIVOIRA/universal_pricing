from scipy.stats import norm
import numpy as np


def bs_call_price_unit(S, K, T, vol):
    normalized_time = vol*np.sqrt(T)
    log_moneyness = np.log(S/K)
    d1 = log_moneyness/normalized_time + 0.5*normalized_time
    d2 = d1 - normalized_time
    N1 = norm.cdf(d1)
    N2 = norm.cdf(d2)
    C = N1*S - N2*K
    return C

def gauss_wave(x, c, b):
    x = x-c
    z = -0.5*(x**2)/b
    g = np.exp(z)
    g = g*((2*b*np.pi)**(-0.5))
    return g

def gaussian_ramp_moment(c, b, k):
    phi = np.exp(c + 0.5*b)
    width = np.sqrt(b)
    d_m = (c - np.log(k))/width
    d_p = d_m+ width
    N_m = norm.cdf(d_m)
    N_p = norm.cdf(d_p)
    psi = phi*N_p - k*N_m
    return psi

def gaussian_ramp_moment_dc(c, b, k):
    phi = np.exp(c + 0.5*b)
    width = np.sqrt(b)
    d_shift = (c - np.log(k)+b)/width
    N_shift = norm.cdf(d_shift)
    dc = phi*N_shift
    return dc

def gaussian_ramp_moment_db(c, b, k):
    phi = np.exp(c + 0.5 * b)
    width = np.sqrt(b)
    d_m = (c - np.log(k)) / width
    d_p = d_m + width
    pdf_p = gauss_wave(d_p*width, 0, b)
    N_p = norm.cdf(d_p)
    db = 0.5*phi*(N_p + pdf_p)
    return db


def blackscholes_to_gaussian(S, K, T, vol):
    k = K/S
    b = T*vol**2
    c = -0.5*b
    return c, b, k
