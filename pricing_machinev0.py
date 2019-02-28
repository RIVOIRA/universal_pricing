import numpy as np
import pandas as pd
import tensorflow as tf
import pathlib
from pricing_machine_utils import *

from LocalVolatility.LocalVolatilityGenerator import *

# folder hierarchy
local_dir = pathlib.Path(__file__).parent
root_dir = local_dir.parent.parent
data_dir = pathlib.Path(root_dir / 'generated_data')
volatility_path = data_dir / 'volatilities.npy'
result_path = data_dir / 'results.npy'
loss_path = data_dir / 'loss.npy'

if not data_dir.exists():
    data_dir.mkdir()

# Constants
vol_scale = 1e-2  # unit is 1/sqrt(year)


# Market Model data
class LocalVolData:
    def __init__(self, maturities, log_strikes, loc_vol_matrix):
        self.log_strikes = log_strikes
        self.maturities = maturities
        self.local_volatilities = loc_vol_matrix

#
class NeuralNetworkTrainingParam:
    def __init__(self, nb_epochs=20, initial_learning_rate=0.01, internal_activation='tanh', batch_size=10):
        self.nb_epochs = nb_epochs
        self.initial_learning_rate = initial_learning_rate
        self.internal_activation = internal_activation
        self.batch_size=batch_size

# Machine learning the local vol to pdf relation
class LocVolToDensityMachine:
    """
    Neural network determining the Gaussian martingale param fitting the best the local vol surface
    in terms of the Dupire gap
    """
    def __init__(self,  local_vol_data, neural_net_param, gaussian_mixture_dim = 1):
        self.log_strikes = local_vol_data.log_strikes
        self.maturities = local_vol_data.maturities
        self.mixture_dim = gaussian_mixture_dim


# Main
def main():
    # Discretization
    strikes = generate_row_interval(val_min=0.95, val_max=1.05, row_length=3)
    log_strikes = np.log(strikes)
    maturities = generate_row_interval(val_min=0.1, val_max=1, row_length=10)

    # Local volatility generation
    loc_vol_generator = UnitLocalVolatilityGenerator(smoothing_parameter=0, gamma=0.25, rng_seed=123456)
    unit_loc_vol_surface = loc_vol_generator.generate(maximal_degree=0)
    scaled_sampled_loc_vol = vol_scale * unit_loc_vol_surface(log_strikes, maturities)

    loc_vol_data = LocalVolData(maturities=maturities,log_strikes=log_strikes, loc_vol_matrix=scaled_sampled_loc_vol)
    np.save(volatility_path, loc_vol_data)

    # Neural network parameters
    nn_param = NeuralNetworkTrainingParam()

    loc_vol_to_pdf = LocVolToDensityMachine(local_vol_data=loc_vol_data, neural_net_param=nn_param, gaussian_mixture_dim=1)



if __name__ == '__main__':
    # To Debug run :
    # run -f has_nan_or_inf
    main()
