from BSpricing import *
import numpy as np


S = 100*np.exp(np.random.randn())
K = 110*np.exp(np.random.randn())
T = 10
vol = 0.12*np.exp(np.random.randn())

C = bs_call_price_unit(S, K, T, vol)

print(C)


c, b, k = blackscholes_to_gaussian(S, K, T, vol)
g = gaussian_ramp_moment(c, b, k)

print('Param')
print('vol = ' + str(vol))
print('S = ' + str(S))
print('K = ' + str(K))

print('c = ' + str(c))
print('b = ' + str(b))
print('k = ' + str(k))


print('Gaussian format now')
print(S*g)

print("Gradient checking now")
d_x = 1e-10

dc = 0.5*(gaussian_ramp_moment(c+d_x, b, k)-gaussian_ramp_moment(c-d_x, b, k))/d_x
print('derivative wrt c')
print(dc)
print('theo')
print(gaussian_ramp_moment_dc(c, b, k))
db = 0.5*(gaussian_ramp_moment(c, b+d_x, k)-gaussian_ramp_moment(c, b-d_x, k))/d_x
print('derivative wrt b')
print(db)
print('theo')
print(gaussian_ramp_moment_db(c, b, k))

b = b + np.exp(np.random.randn())
print('new b = ' + str(b))
print(gaussian_ramp_moment_dc(c, b, k))
db = 0.5*(gaussian_ramp_moment(c, b+d_x, k)-gaussian_ramp_moment(c, b-d_x, k))/d_x
print('derivative wrt b')
print(db)
print('theo')
print(gaussian_ramp_moment_db(c, b, k))