""" Neural network activation functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# Contributor : Ankit Grover
# License: BSD 3 clause
from mlrose_hiive.decorators import short_name

import numpy as np

import warnings
warnings.filterwarnings("ignore")


@short_name('leaky_relu')
def leaky_relu(x, alpha=0.3, deriv=False):
    """ Leaky ReLU activation function

    Parameters
    ----------
    x: array
        Array containing input data.
    alpha: int , default : 0.3
        Alpha value to be set for applying small negative gradient
    deriv: bool, default: False
        Whether to return the function or its derivative.
        Set True for derivative.

    Returns
    -------
    fx: array
        Value of activation function at x
    """
    fx = np.copy(x)
    fx = np.where(fx < 0, fx * alpha, fx)

    if deriv:
        fx[np.where(fx > 0)] = 1
        fx[np.where(fx < 0)] = alpha

    return fx
