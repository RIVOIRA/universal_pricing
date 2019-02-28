from BSpricing import bs_call_price_unit, gauss_wave
from GaussianMartingaleMixture import *
import numpy as np

# input data
s0 = 117
k = 100
vol = np.sqrt(2)
t = 3
b = t * (vol ** 2)
gamma = 0.5

# omega function
print('omega function')
print(omega(k, b, gamma=gamma, s0=s0))
print(bs_call_price_unit(s0 * np.exp(gamma), k, t, vol))

# gauss_exp_martingale_pdf
print('gauss_exp_martingale_pdf')
print(gauss_exp_martingale_pdf(k, b, gamma, s0))

# d_omega_db
print('d_omega_db')
db = 1e-5
delta_omega_deltab = (omega(k, b + db, gamma, s0) - omega(k, b - db, gamma, s0)) / (2 * db)

print(str(delta_omega_deltab)+' <--grad check now')
print(d_omega_db(k, b, gamma, s0))
print(0.5*k * gauss_wave(log_norm_dist(gamma, s0, k, b), 0, b))

