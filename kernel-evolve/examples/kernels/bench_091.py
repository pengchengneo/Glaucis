"""Auto-generated Glaucis template for L1 problem 91: cumsum_reverse."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 32768
input_shape = (32768,)
dim = 1


class Model:
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.
    """
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        return jnp.flip(jnp.cumsum(jnp.flip(x, axis=self.dim), axis=self.dim), axis=self.dim)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]


def get_init_inputs():
    return [dim]


# EVOLVE-BLOCK-START
def optimized_compute(_id=91):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
