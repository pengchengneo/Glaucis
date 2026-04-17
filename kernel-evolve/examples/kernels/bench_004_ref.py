"""Auto-generated Glaucis reference for L1 problem 4: Matrix_vector_multiplication_."""

import jax
import jax.numpy as jnp


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


def simple_compute(_id=4):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
