from General2DSurface import General2DTaylorSurface
import numpy as np
import math


class LocalVolatilitySurface(General2DTaylorSurface):
    def __init__(self, derivatives_at_origin, ref_point, gamma):
        General2DTaylorSurface.__init__(self, derivatives_at_origin, ref_point)
        self.global_exp_scale = gamma
        pass

    def __call__(self, x, t):
        t = General2DTaylorSurface.format_inputs(t)
        y = np.sqrt(t)
        log_vol = super(LocalVolatilitySurface, self).__call__(x, y)
        gamma = self.global_exp_scale
        return np.exp(gamma * log_vol)


class UnitLocalVolatilityGenerator:
    def __init__(self, smoothing_parameter, gamma, rng_seed=None):
        self.smoothing_parameter = smoothing_parameter
        self.global_exp_scale = gamma
        if rng_seed is not None:
            np.random.RandomState(rng_seed)
        self.initial_random_state = np.random.get_state()
        pass

    def reset_random(self):
        np.random.set_state(self.initial_random_state)
        pass

    def get_random_state(self):
        return self.initial_random_state

    def generate_derivatives(self, maximal_degree):
        ordered_derivatives = dict()
        alpha = self.smoothing_parameter
        for n in range(0, maximal_degree + 1):
            a = np.random.normal(0., 1., n + 1)
            w = np.exp(-alpha * (n ** 2))
            a = w * a
            ordered_derivatives[n] = a
        return ordered_derivatives

    @staticmethod
    def generate_random_taylor_origin():
        c_x = np.random.uniform(-1, 1)
        c_y = np.random.uniform(0, 1)
        return np.array((c_x, c_y))

    def generate(self, maximal_degree):
        ordered_derivatives = self.generate_derivatives(maximal_degree)
        taylor_origin = UnitLocalVolatilityGenerator.generate_random_taylor_origin()
        gamma = self.global_exp_scale
        local_vol_surface = LocalVolatilitySurface(ordered_derivatives, taylor_origin, gamma)
        return local_vol_surface
