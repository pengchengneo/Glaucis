"""Auto-generated Glaucis template for L1 problem 43: Max_Pooling_3D."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 16
channels = 32
dim1 = 128
dim2 = 128
dim3 = 128
kernel_size = 3
stride = 2
padding = 1
dilation = 3


class Model:
    """
    Simple model that performs Max Pooling 3D.
    """
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x):
        # x shape: (batch_size, channels, dim1, dim2, dim3)
        padding_config = [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)]
        return jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max,
            window_dimensions=(1, 1, self.kernel_size, self.kernel_size, self.kernel_size),
            window_strides=(1, 1, self.stride, self.stride, self.stride),
            padding=padding_config,
            window_dilation=(1, 1, self.dilation, self.dilation, self.dilation)
        )


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, channels, dim1, dim2, dim3))
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]


# EVOLVE-BLOCK-START
def optimized_compute(_id=43):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
