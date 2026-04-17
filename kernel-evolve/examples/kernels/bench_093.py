"""Auto-generated Glaucis template for L1 problem 93: masked_cumsum."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 32768
input_shape = (32768,)
dim = 1


class Model:
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.
    """
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x, mask):
        return jnp.cumsum(x * mask, axis=self.dim)


def get_inputs():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1, shape=(batch_size, *input_shape))
    mask = jax.random.randint(k2, shape=x.shape, minval=0, maxval=2).astype(jnp.bool_)
    return [x, mask]


def get_init_inputs():
    return [dim]


# EVOLVE-BLOCK-START
def optimized_compute(_id=93):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
