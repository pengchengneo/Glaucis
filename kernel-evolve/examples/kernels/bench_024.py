"""Auto-generated Glaucis template for L1 problem 24: LogSoftmax."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 4096
dim = 393216


class Model:
    """
    Simple model that performs a LogSoftmax activation.
    """
    def __init__(self, dim: int = 1):
        self.dim = dim

    def __call__(self, x):
        """
        Applies LogSoftmax activation to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, dim).

        Returns:
            jnp.ndarray: Output tensor with LogSoftmax applied, same shape as input.
        """
        return jax.nn.log_softmax(x, axis=self.dim)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed


# EVOLVE-BLOCK-START
def optimized_compute(_id=24):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
