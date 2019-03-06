import numpy as np


class MartingaleProbabilitiesNumpy:
    def __init__(self, positions, expected_center):
        self.expected_center = expected_center
        self.dimension = len(positions)
        positions = np.reshape(positions, [self.dimension, 1])
        self.index_above = (positions > expected_center)
        self.index_below = ~self.index_above
        self.positions_above = positions[self.index_above[:, 0]]
        self.positions_below = positions[self.index_below[:, 0]]

    @staticmethod
    def total_mass_and_momentum(x, m):
        mass = np.sum(m, axis=0, keepdims=True)
        momentum = np.sum(m * x, axis=0, keepdims=True)
        return mass, momentum

    def format_inputs(self, masses_or_positions):
        masses_or_positions = np.reshape(masses_or_positions, (self.dimension, -1))
        return masses_or_positions

    def __call__(self, masses):
        masses = self.format_inputs(masses)
        masses_above = masses[self.index_above[:, 0]]
        aggr_mass_above, momentum_above = MartingaleProbabilitiesNumpy.total_mass_and_momentum(self.positions_above,
                                                                                               masses_above)
        center_above = momentum_above / aggr_mass_above

        masses_below = masses[self.index_below[:, 0]]
        aggr_mass_below, momentum_below = MartingaleProbabilitiesNumpy.total_mass_and_momentum(self.positions_below,
                                                                                               masses_below)
        center_below = momentum_below / aggr_mass_below
        theta_above = (self.expected_center - center_below) / (center_above - center_below)

        theta_below = 1 - theta_above
        scaling_below = theta_below / aggr_mass_below
        scaling_above = theta_above / aggr_mass_above
        probabilities = masses
        probabilities[self.index_above[:, 0]] *= scaling_above
        probabilities[self.index_below[:, 0]] *= scaling_below

        return probabilities
