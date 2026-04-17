"""Auto-generated Glaucis template for L1 problem 96: HuberLoss."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 32768
input_shape = (32768,)
dim = 1


class Model:
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.
    """
    def __init__(self):
        pass

    def __call__(self, predictions, targets):
        diff = predictions - targets
        abs_diff = jnp.abs(diff)
        loss = jnp.where(abs_diff < 1.0, 0.5 * diff ** 2, abs_diff - 0.5)
        return jnp.mean(loss)


def get_inputs():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    scale = jax.random.uniform(k1, shape=())
    return [jax.random.uniform(k2, shape=(batch_size, *input_shape)) * scale, jax.random.uniform(k3, shape=(batch_size, *input_shape))]


def get_init_inputs():
    return []


# EVOLVE-BLOCK-START
def optimized_compute(_id=96):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
