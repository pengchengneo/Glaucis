"""Auto-generated Glaucis template for L1 problem 89: cumsum."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 32768
input_shape = (32768,)
dim = 1


class Model:
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        self.dim = dim

    def __call__(self, x):
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x: Input tensor of shape (batch_size, *input_shape), where `*input_shape`
               can vary depending on the use case.

        Returns:
            Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        return jnp.cumsum(x, axis=self.dim)


def get_inputs():
    """
    Generates random inputs for testing the Scan model.

    Returns:
        list: A list containing a single randomly generated tensor with shape
              (batch_size, *input_shape).
    """
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]


def get_init_inputs():
    """
    Returns the initialization parameters for the Scan model.

    Returns:
        list: A list containing the `dim` parameter for model initialization.
    """
    return [dim]


# EVOLVE-BLOCK-START
def optimized_compute(_id=89):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
