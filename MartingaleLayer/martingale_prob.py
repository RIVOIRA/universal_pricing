import numpy as np
import tensorflow as tf
# from debug_utils import debug
from tensorflow.python.keras.layers import Lambda


class MartingaleProbabilities:
    """
    Implements a functor class mapping positive numbers to probabilities having some predefined expectation

    Arguments:
    sorted_positions -- sorted vector of the positions of the masses, shape (dimension, 1)
    expected_center --  single real number, specifying the expected value of the positions
                        requirement -- min(positions) < expected_center < max(positions)
    main method:
        __call__(m):
                    Arguments -- vector of positive masses, shape (dimension, 1)
                    Returns -- vector of probabilities of shape (dimension, 1) with an expectation = expected_center
    Example:
    x =  [[0.03691922]
         [0.04345877]
         [0.07926783]
         [0.11754176]
         [0.14145406]
         [0.33845356]
         [0.5029091 ]
         [0.7426342 ]
         [0.81837039]
         [0.84120082]]

    m = [[0.19470918 0.91079298]
         [0.05973249 0.12383036]
         [0.05228813 0.72180973]
         [0.15716975 0.95096828]
         [0.92916646 0.20439648]
         [0.79720128 0.66321122]
         [0.7209362  0.6647576 ]
         [0.76593861 0.99214558]
         [0.84298503 0.11994978]
         [0.81391226 0.67759247]]

    mu = 0.6660650695881166

    martingale_prob = MartingaleProbabilities(sorted_positions=x, expected_center=mu)
    M = tf.placeholder(tf.float64, shape=m.shape)
    martingale_prob_tensor = martingale_prob(M)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        p = martingale_prob_tensor.eval(session=sess,
                                        feed_dict={M: m})
    p = [[0.01721202 0.04278179]
         [0.00528027 0.00581656]
         [0.0046222  0.03390487]
         [0.01389358 0.04466891]
         [0.082137   0.00960092]
         [0.07047146 0.03115237]
         [0.06372974 0.03122501]
         [0.23477743 0.44396534]
         [0.25839389 0.05367513]
         [0.24948243 0.3032091 ]]
    """

    def __init__(self, sorted_positions, expected_center):
        self.expected_center = expected_center
        self.dimension = max(sorted_positions.shape)
        positions = np.reshape(sorted_positions, [self.dimension, 1])
        positions = np.sort(positions, axis=0)
        index_just_above = np.searchsorted(positions[:, 0], expected_center)
        self.index_just_above = index_just_above[0]
        self.positions_below = positions[:self.index_just_above]
        self.positions_above = positions[self.index_just_above:]

    @staticmethod
    def aggregated_mass_and_momentum(x, m):
        aggr_mass = tf.reduce_sum(m, axis=0)
        momentum = tf.reduce_sum(x * m, axis=0)
        return aggr_mass, momentum

    def __call__(self, masses):
        masses = tf.reshape(masses, (self.dimension, -1))

        masses_below = masses[: self.index_just_above, :]
        aggr_mass_below, momentum_below = MartingaleProbabilities.aggregated_mass_and_momentum(self.positions_below,
                                                                                               masses_below)
        center_below = momentum_below / aggr_mass_below

        masses_above = masses[self.index_just_above:, :]
        aggr_mass_above, momentum_above = MartingaleProbabilities.aggregated_mass_and_momentum(self.positions_above,
                                                                                               masses_above)
        center_above = momentum_above / aggr_mass_above

        theta_above = (self.expected_center - center_below) / (center_above - center_below)
        theta_below = 1 - theta_above

        scaling_below = theta_below / aggr_mass_below
        scaling_above = theta_above / aggr_mass_above

        prob_below = tf.matmul(masses_below, tf.diag(scaling_below))
        prob_above = tf.matmul(masses_above, tf.diag(scaling_above))
        probabilities = tf.concat([prob_below, prob_above], axis=0)

        return probabilities
