"""Auto-generated Glaucis template for L1 problem 37: FrobeniusNorm_."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 112
features = 64
dim1 = 512
dim2 = 512


class Model:
    """
    Simple model that performs Frobenius norm normalization.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        norm = jnp.sqrt(jnp.sum(x ** 2))
        return x / norm


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]


def get_init_inputs():
    return []


# EVOLVE-BLOCK-START
def optimized_compute(_id=37):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
