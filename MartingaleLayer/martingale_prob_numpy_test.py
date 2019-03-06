import numpy as np
from martingale_prob_numpy import *

x = np.random.uniform(0, 1, 10)
m = np.random.uniform(0, 1, (10, 2))

mu = np.random.uniform(0.1, 0.9, 1)
martingale_prob = MartingaleProbabilitiesNumpy(positions=x, expected_center=mu)
p = martingale_prob(m)
print(p)
print(x)
print('Expected center =' + str(mu[0]))
x = martingale_prob.format_inputs(x)
print('moment = ' + str(np.sum(x * p, axis=0)))
print('Total probability = ' + str(np.sum(p, axis=0)))
