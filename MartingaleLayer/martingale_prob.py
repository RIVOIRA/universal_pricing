import numpy as np
import tensorflow as tf


class MartingaleProbabilities:
    def __init__(self, positions, expected_center):
        self.expected_center = expected_center
        self.dimension = len(positions)
        positions = tf.reshape(positions, [self.dimension, 1])
        self.index_above = (positions > expected_center)
        self.index_below = ~self.index_above
        print(self.index_above)
        self.positions_above = tf.boolean_mask(positions, self.index_above)
        self.positions_below = tf.boolean_mask(positions, self.index_below)

    @staticmethod
    def aggregated_mass_and_momentum(x, m):
        mass = tf.reduce_sum(m)
        momentum = tf.reduce_sum(m * x)
        return mass, momentum

    def __call__(self, masses):
        masses = tf.reshape(masses, [self.dimension, -1])
        masses_above = tf.boolean_mask(masses, self.index_above)
        aggr_mass_above, momentum_above = MartingaleProbabilities.aggregated_mass_and_momentum(self.positions_above,
                                                                                               masses_above)
        center_above = momentum_above / aggr_mass_above

        masses_below = tf.boolean_mask(masses, self.index_below)
        aggr_mass_below, momentum_below = MartingaleProbabilities.aggregated_mass_and_momentum(self.positions_below,
                                                                                               masses_below)
        center_below = momentum_below / aggr_mass_below
        theta_above = (expectation - center_below) / (center_above - center_below)
        theta_below = 1 - theta_above
        scaling_below = theta_below / aggr_mass_below * tf.ones_like(tf.cast(self.index_below, tf.float64))
        scaling_above = theta_above / aggr_mass_above * tf.ones_like(masses)
        scaling_factor = tf.where(self.index_below, scaling_below, scaling_above)

        probabilities = masses * scaling_factor

        return probabilities


x = np.random.uniform(0, 1, 10)
m = np.random.uniform(0, 1, 10)
# x = np.arange(10)
# m = np.ones(10)
expectation = 0.5
index_above = (x > expectation)
index_below = ~index_above
print(x)
print(index_above)
print(x[index_above])
print(x[index_below])


def mass_and_momentum(x, m):
    mass = np.sum(m)
    momentum = np.sum(x * m)
    return mass, momentum


x_above = x[index_above]
masses_above = m[index_above]
mass_above, momentum_above = mass_and_momentum(x_above, masses_above)
print('mass above =' + str(mass_above))
print('momentum above =' + str(momentum_above))
center_above = momentum_above / mass_above
print('above center = ' + str(center_above))

x_below = x[index_below]
masses_below = m[index_below]
mass_below, momentum_below = mass_and_momentum(x_below, masses_below)
print('mass below = ' + str(mass_below))
print('momentum below =' + str(momentum_below))

center_below = momentum_below / mass_below
print('below center = ' + str(center_below))

theta_above = (expectation - center_below) / (center_above - center_below)
theta_below = 1 - theta_above
print('theta')
print(theta_above)
p = m
p[index_above] = theta_above * p[index_above] / mass_above
p[index_below] = theta_below * p[index_below] / mass_below

print(p)
print(sum(p))
print(np.sum(x * p))

martingale_prob = MartingaleProbabilities(positions=x, expected_center=expectation)

proba = martingale_prob(m)
print(p)
sess = tf.Session()
print(sess.run(proba))
