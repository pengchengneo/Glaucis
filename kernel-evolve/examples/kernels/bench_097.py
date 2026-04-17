"""Auto-generated Glaucis template for L1 problem 97: ScaledDotProductAttention."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 32
num_heads = 32
sequence_length = 512
embedding_dimension = 1024


class Model:
    """
    A model that computes Scaled Dot-Product Attention.
    """
    def __init__(self):
        pass

    def __call__(self, Q, K, V):
        d_k = Q.shape[-1]
        scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / jnp.sqrt(float(d_k))
        attn_weights = jax.nn.softmax(scores, axis=-1)
        return jnp.matmul(attn_weights, V)


def get_inputs():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    Q = jax.random.uniform(k1, shape=(batch_size, num_heads, sequence_length, embedding_dimension))
    K = jax.random.uniform(k2, shape=(batch_size, num_heads, sequence_length, embedding_dimension))
    V = jax.random.uniform(k3, shape=(batch_size, num_heads, sequence_length, embedding_dimension))
    return [Q, K, V]


def get_init_inputs():
    return []


# EVOLVE-BLOCK-START
def optimized_compute(_id=97):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
