"""Auto-generated Glaucis template for L1 problem 20: LeakyReLU."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 4096
dim = 393216


class Model:
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        self.negative_slope = negative_slope

    def __call__(self, x):
        """
        Applies LeakyReLU activation to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor of any shape.

        Returns:
            jnp.ndarray: Output tensor with LeakyReLU applied, same shape as input.
        """
        return jax.nn.leaky_relu(x, negative_slope=self.negative_slope)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed


# EVOLVE-BLOCK-START
def optimized_compute(_id=20):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
