import numpy as np
from martingale_prob_numpy import *
# from debug_utils import debug
from martingale_prob import *
import tensorflow as tf

x = np.random.uniform(0, 1, 10)
x = np.reshape(x, [10, 1])
x = np.sort(x, axis=0)

m = np.random.uniform(0, 1, (10, 2))

mu = np.random.uniform(0.1, 0.9, 1)
martingale_prob = MartingaleProbabilities(sorted_positions=x, expected_center=mu)
M = tf.placeholder(tf.float64, shape=m.shape)
martingale_prob_tensor = martingale_prob(M)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    p = martingale_prob_tensor.eval(session=sess,
                                    feed_dict={M: m})

print('x =' + str(x))
print('m =' + str(m))
print('p =' + str(p))

print('Expected center =' + str(mu[0]))
print('moment = ' + str(np.sum(np.multiply(x.reshape((len(x), 1)), p), axis=0)))
print('Total probability = ' + str(np.sum(p, axis=0)))

print('With Numpy now')
martingale_prob = MartingaleProbabilitiesNumpy(positions=x, expected_center=mu)
p = martingale_prob(m)
print('p =' + str(p))
