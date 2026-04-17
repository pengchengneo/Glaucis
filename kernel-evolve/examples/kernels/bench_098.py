"""Auto-generated Glaucis template for L1 problem 98: KLDivLoss."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1


class Model:
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.
    """
    def __init__(self):
        pass

    def __call__(self, predictions, targets):
        log_predictions = jnp.log(predictions)
        loss = targets * (jnp.log(targets) - log_predictions)
        return jnp.sum(loss) / predictions.shape[0]


def get_inputs():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    scale = jax.random.uniform(k1, shape=())
    return [jax.nn.softmax(jax.random.uniform(k2, shape=(batch_size, *input_shape)) * scale, axis=-1),
            jax.nn.softmax(jax.random.uniform(k3, shape=(batch_size, *input_shape)), axis=-1)]


def get_init_inputs():
    return []


# EVOLVE-BLOCK-START
def optimized_compute(_id=98):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
