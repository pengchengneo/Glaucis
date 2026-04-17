"""Auto-generated Glaucis reference for L1 problem 1: Square_matrix_multiplication_."""

import jax
import jax.numpy as jnp


N = 2048 * 2


class Model:
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A: Input matrix A of shape (N, N).
            B: Input matrix B of shape (N, N).

        Returns:
            Output matrix C of shape (N, N).
        """
        return jnp.matmul(A, B)


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(N, N))
    B = jax.random.uniform(key2, shape=(N, N))
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed


def simple_compute(_id=1):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
