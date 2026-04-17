"""Auto-generated Glaucis template for L1 problem 3: Batched_matrix_multiplication."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4


class Model:
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return jnp.matmul(A, B)


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(batch_size, m, k))
    B = jax.random.uniform(key2, shape=(batch_size, k, n))
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed


# EVOLVE-BLOCK-START
def optimized_compute(_id=3):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
