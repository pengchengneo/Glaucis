"""Auto-generated Glaucis template for L1 problem 5: Matrix_scalar_multiplication."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


M = 16384 * 4
N = 4096 * 4


class Model:
    """
    Simple model that performs a matrix-scalar multiplication (C = A * s)
    """
    def __init__(self):
        pass

    def __call__(self, A, s):
        """
        Performs matrix-scalar multiplication.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return A * s


def get_inputs():
    key = jax.random.PRNGKey(0)
    A = jax.random.uniform(key, shape=(M, N))
    s = 3.14
    return [A, s]


def get_init_inputs():
    return []  # No special initialization inputs needed


# EVOLVE-BLOCK-START
def optimized_compute(_id=5):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
