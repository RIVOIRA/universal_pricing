import numpy as np
from scipy.stats import norm


def log_norm_dist(gamma, s0, k, b):
    """
    Compute the corrected log_moneyness:
    \delta(k,t) = \gamma + \ln(s0/K) - b(t)/2
    :param gamma: log distance position of the centroid relative to s0
    :param s0: spot price of the stock
    :param k: strike price
    :param b: variance of the gaussian component
    :return: gamma + log(s0/k) -0.5*b
    """
    return gamma + np.log(s0 / k) - 0.5 * b


def omega(k, b, gamma, s0):
    """
    Price of the call C in the Gaussian exponential martingale framework of variance b and initial mean gamma
        C = s0*exp(gamma)*N(d+)-K*N(d-)
    See Omega function of the documentation
    :param k: strike price
    :param b: Gaussian variance
    :param gamma: Related to the constant expected price s_mean by gamma = log(s_mean/s0)
    :param s0: spot price of the stock
    :return: call price under the Gaussian martingale assumption

    Example:
    s0 = 117
    k = 100
    vol = np.sqrt(2)
    t = 3
    b = t * (vol ** 2)
    gamma = 0.5
    print(omega(k, b, gamma=gamma, s0=s0))      # 104.2483849182531
    """
    sqrt_b = np.sqrt(b)
    d_minus = log_norm_dist(gamma, s0, k, b) / sqrt_b
    cdf_minus = norm.cdf(d_minus)
    d_plus = d_minus + sqrt_b
    cdf_plus = norm.cdf(d_plus)
    return s0 * np.exp(gamma) * cdf_plus - k * cdf_minus


def centered_gauss_pdf(x, dx2):
    """
    probability density function of the gaussian variable of variance dx2:
    p(x) = exp(-0.5*x^2/dx2)/sqrt(2*pi*dx2)
    :param x: point at which the density must be evaluated
    :param dx2: variance of the Gaussian centered variable
    :return: probability density of being around x
    """
    z = -0.5 * (x ** 2) / dx2
    g = np.exp(z)
    g = g * ((2 * dx2 * np.pi) ** (-0.5))
    return g


def gauss_exp_martingale_pdf(k, b, gamma, s0):
    """
    probability density function of the Gaussian exponential martingale
        C = s0*exp(gamma)*N(d+)-K*N(d-)
    See Omega function of the documentation
    :param k: strike price
    :param b: Gaussian variance
    :param gamma: Related to the constant expected price s_mean by gamma = log(s_mean/s0)
    :param s0: spot price of the stock
    :return: call price under the Gaussian martingale assumption

    Example:
    s0 = 117
    k = 100
    vol = np.sqrt(2)
    t = 3
    b = t * (vol ** 2)
    gamma = 0.5
    print(gauss_exp_martingale_pdf(k, b, gamma, s0))     # 0.10307611800693516
    """
    return centered_gauss_pdf(log_norm_dist(gamma, s0, k, b), b)


def d_omega_db(k, b, gamma, s0):
    """
    first order derivative of omega wrt to b
    :param k: strike price
    :param b: Gaussian variance
    :param gamma: Related to the constant expected price s_mean by gamma = log(s_mean/s0)
    :param s0: spot price of the stock
    :return: call price under the Gaussian martingale assumption

    Example:
    s0 = 117
    k = 100
    vol = np.sqrt(2)
    t = 3
    b = t * (vol ** 2)
    gamma = 0.5
    print(omega(k, b, gamma=gamma, s0=s0))      # 5.153805900346757
    """
    return 0.5 * k * gauss_exp_martingale_pdf(k, b, gamma, s0)
