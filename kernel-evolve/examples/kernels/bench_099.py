"""Auto-generated Glaucis template for L1 problem 99: TripletMarginLoss."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 32768
input_shape = (8192,)
dim = 1


class Model:
    """
    A model that computes Triplet Margin Loss for metric learning tasks.
    """
    def __init__(self, margin=1.0):
        self.margin = margin

    def __call__(self, anchor, positive, negative):
        dist_pos = jnp.sqrt(jnp.sum((anchor - positive) ** 2, axis=-1))
        dist_neg = jnp.sqrt(jnp.sum((anchor - negative) ** 2, axis=-1))
        loss = jnp.mean(jnp.maximum(dist_pos - dist_neg + self.margin, 0.0))
        return loss


def get_inputs():
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    scale = jax.random.uniform(k1, shape=())
    return [jax.random.uniform(k2, shape=(batch_size, *input_shape)) * scale,
            jax.random.uniform(k3, shape=(batch_size, *input_shape)),
            jax.random.uniform(k4, shape=(batch_size, *input_shape))]


def get_init_inputs():
    return [1.0]


# EVOLVE-BLOCK-START
def optimized_compute(_id=99):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
