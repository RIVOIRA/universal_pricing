import numpy as np
import scipy
from scipy import misc


class General2DTaylorSurface:
    """
    Handles a two variate real function F from its Taylor expansion around a reference point p_0 =(x_0,y_0)
    stored into dictionary of vectors
        element 0 is the value of the Polynomial at 0: F(p_0)
        element 1 is the derivative wrt first and second variables: DxF(x_0) DyF(p_0)
        element 2 is made up of all second order derivatives: DxxF(p_0) DxyF(p_0) DyyF(p_0)
        ...
        element n is made up of all nth derivatives and must be of size n+1

    Example:
        d = dict()
        d[0] = 3
        d[1] = [1, 4]
        d[2] = [-1, 2, -5]

        mySurface = General2DTaylorSurface(d)
        x = np.linspace(0., 1, 2)
        y = np.linspace(0., 1, 3)
        print(mySurface(x, y))

        Output:
        [[3.    4.375  4.5  ]
         [3.5   5.875  7.   ]]

    """

    def __init__(self, derivatives_at_origin, ref_point=np.zeros((2, 1))):
        self.nb_orders = len(derivatives_at_origin)
        self.taylor_origin = ref_point
        poly_coef = dict()
        for n in range(0, self.nb_orders):
            ordered_der = derivatives_at_origin[n]
            factorials = scipy.misc.factorial(np.arange(n + 1))
            binomial_factor = np.multiply(factorials, np.flip(factorials))
            ordered_der = np.divide(ordered_der, binomial_factor)
            poly_coef[n] = ordered_der
        self.polyCoefficients = poly_coef
        pass

    @staticmethod
    def format_inputs(x):
        n_x = len(x)
        x = x.reshape((n_x, 1))
        x = np.array(x)
        return x

    def centered_formatted_inputs(self, x, y):
        formatted_inputs = dict()
        p_0 = self.taylor_origin
        x_0 = p_0[0]
        formatted_x = General2DTaylorSurface.format_inputs(x)
        formatted_x = formatted_x - x_0
        formatted_inputs[0] = formatted_x
        y_0 = p_0[1]
        formatted_y = General2DTaylorSurface.format_inputs(y)
        formatted_y = formatted_y - y_0
        formatted_inputs[1] = formatted_y
        return formatted_inputs

    def __call__(self, x_raw, y_raw):
        formatted_coordinates = self.centered_formatted_inputs(x_raw, y_raw)
        x = formatted_coordinates[0]
        y = formatted_coordinates[1]
        input_size = (len(x), len(y))
        poly_val = self.polyCoefficients[0] * np.ones(input_size)
        for n in range(1, self.nb_orders):
            poly_coef_n = self.polyCoefficients[n]
            for k in range(0, n + 1):
                d_poly_val = poly_coef_n[k] * np.power(x, n - k) * (np.power(y, k)).T
                poly_val = poly_val + d_poly_val
        return poly_val


