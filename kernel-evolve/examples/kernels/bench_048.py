"""Auto-generated Glaucis template for L1 problem 48: Mean_reduction_over_a_dimension."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 128
dim1 = 4096
dim2 = 4095


class Model:
    """
    Simple model that performs mean reduction over a specific dimension.
    """
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        return jnp.mean(x, axis=self.dim)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim1, dim2))
    return [x]


def get_init_inputs():
    return [1]


# EVOLVE-BLOCK-START
def optimized_compute(_id=48):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
