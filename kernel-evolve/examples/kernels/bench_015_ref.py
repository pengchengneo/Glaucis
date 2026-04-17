"""Auto-generated Glaucis reference for L1 problem 15: Matmul_for_lower_triangular_matrices."""

import jax
import jax.numpy as jnp


M = 4096


class Model:
    """
    Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices.
    """
    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (jnp.ndarray): Lower triangular matrix of shape (N, N).
            B (jnp.ndarray): Lower triangular matrix of shape (N, N).

        Returns:
            jnp.ndarray: The result of matrix multiplication C of shape (N, N).
        """
        return jnp.tril(jnp.matmul(A, B))


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jnp.tril(jax.random.uniform(key1, shape=(M, M)))
    B = jnp.tril(jax.random.uniform(key2, shape=(M, M)))
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed


def simple_compute(_id=15):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
