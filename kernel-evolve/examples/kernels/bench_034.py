"""Auto-generated Glaucis template for L1 problem 34: InstanceNorm."""

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
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps

    def __call__(self, x):
        # x shape: (batch_size, num_features, height, width)
        mean = jnp.mean(x, axis=(2, 3), keepdims=True)
        var = jnp.var(x, axis=(2, 3), keepdims=True)
        return (x - mean) / jnp.sqrt(var + self.eps)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]


def get_init_inputs():
    return [features]


# EVOLVE-BLOCK-START
def optimized_compute(_id=34):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
