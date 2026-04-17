"""Auto-generated Glaucis template for L1 problem 10: 3D_tensor_matrix_multiplication."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


N = 16
M = 1024
K = 2048
L = 768


class Model:
    """
    Performs 3D tensor-matrix multiplication.
    """
    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (jnp.ndarray): Input 3D tensor of shape (N, M, K).
            B (jnp.ndarray): Input matrix of shape (K, L).

        Returns:
            jnp.ndarray: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return jnp.matmul(A, B)


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(N, M, K))
    B = jax.random.uniform(key2, shape=(K, L))
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed


# EVOLVE-BLOCK-START
def optimized_compute(_id=10):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)
# EVOLVE-BLOCK-END
