"""Auto-generated Glaucis template for L1 problem 90: cumprod."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 32768
input_shape = (32768,)
dim = 1


class Model:
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        self.dim = dim

    def __call__(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x: Input tensor of shape (batch_size, *input_shape).

        Returns:
            Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return jnp.cumprod(x, axis=self.dim)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]


def get_init_inputs():
    return [dim]


# EVOLVE-BLOCK-START
def optimized_compute(_id=90):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
