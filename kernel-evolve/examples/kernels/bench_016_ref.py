"""Auto-generated Glaucis reference for L1 problem 16: Matmul_with_transposed_A."""

import jax
import jax.numpy as jnp


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2


class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B)
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
        return jnp.matmul(A.T, B)


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(K, M))
    B = jax.random.uniform(key2, shape=(K, N))
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed


def simple_compute(_id=16):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
