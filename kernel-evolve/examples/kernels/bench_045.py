"""Auto-generated Glaucis template for L1 problem 45: Average_Pooling_2D."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 16
channels = 64
height = 2048
width = 2048
kernel_size = 11


class Model:
    """
    Simple model that performs 2D Average Pooling.
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def __call__(self, x):
        # x shape: (batch_size, channels, height, width)
        padding_config = [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)]
        sums = jax.lax.reduce_window(
            x, 0.0, jax.lax.add,
            window_dimensions=(1, 1, self.kernel_size, self.kernel_size),
            window_strides=(1, 1, self.stride, self.stride),
            padding=padding_config
        )
        return sums / (self.kernel_size * self.kernel_size)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, channels, height, width))
    return [x]


def get_init_inputs():
    return [kernel_size]


# EVOLVE-BLOCK-START
def optimized_compute(_id=45):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
