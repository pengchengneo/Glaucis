"""Auto-generated Glaucis template for L1 problem 100: HingeLoss."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 32768
input_shape = (32768,)
dim = 1


class Model:
    """
    A model that computes Hinge Loss for binary classification tasks.
    """
    def __init__(self):
        pass

    def __call__(self, predictions, targets):
        return jnp.mean(jnp.maximum(1 - predictions * targets, 0))


def get_inputs():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    return [jax.random.uniform(k1, shape=(batch_size, *input_shape)),
            jax.random.randint(k2, shape=(batch_size,), minval=0, maxval=2).astype(jnp.float32) * 2 - 1]


def get_init_inputs():
    return []


# EVOLVE-BLOCK-START
def optimized_compute(_id=100):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
