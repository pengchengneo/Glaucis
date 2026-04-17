"""Auto-generated Glaucis template for L1 problem 8: Matmul_with_irregular_shapes_."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


M = 8205
K = 2949
N = 5921


class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) with irregular shapes
    """
    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Performs matrix multiplication of A and B.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        return jnp.matmul(A, B)


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(M, K))
    B = jax.random.uniform(key2, shape=(K, N))
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed


# EVOLVE-BLOCK-START
def optimized_compute(_id=8):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
