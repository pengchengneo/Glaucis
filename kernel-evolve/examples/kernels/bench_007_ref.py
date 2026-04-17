"""Auto-generated Glaucis reference for L1 problem 7: Matmul_with_small_K_dimension_."""

import jax
import jax.numpy as jnp


M = 16384 * 2
N = 16384 * 2
K = 32 * 2


class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """
    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
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


def simple_compute(_id=7):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
