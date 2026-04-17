"""Auto-generated Glaucis template for L1 problem 4: Matrix_vector_multiplication_."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


M = 256 * 8  # 2048
K = 131072 * 8  # 1048576


class Model:
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return jnp.matmul(A, B)


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(M, K))
    B = jax.random.uniform(key2, shape=(K, 1))
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed


# EVOLVE-BLOCK-START
def optimized_compute(_id=4):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
