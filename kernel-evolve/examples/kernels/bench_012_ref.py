"""Auto-generated Glaucis reference for L1 problem 12: Matmul_with_diagonal_matrices_."""

import jax
import jax.numpy as jnp


M = 4096
N = 4096


class Model:
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (jnp.ndarray): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (jnp.ndarray): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            jnp.ndarray: The result of the matrix multiplication. Shape: (N, M).
        """
        # Logically equivalent to jnp.diag(A) @ B
        # more efficient as no need to materialize a full N×N matrix
        return A[:, None] * B


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(N,))
    B = jax.random.uniform(key2, shape=(N, M))
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed


def simple_compute(_id=12):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
