"""Auto-generated Glaucis reference for L1 problem 14: Matmul_for_upper_triangular_matrices."""

import jax
import jax.numpy as jnp


N = 4096


class Model:
    """
    Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """
    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            A (jnp.ndarray): Upper triangular matrix of shape (N, N).
            B (jnp.ndarray): Upper triangular matrix of shape (N, N).

        Returns:
            jnp.ndarray: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return jnp.triu(jnp.matmul(A, B))


def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jnp.triu(jax.random.uniform(key1, shape=(N, N)))
    B = jnp.triu(jax.random.uniform(key2, shape=(N, N)))
    return [A, B]


def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []


def simple_compute(_id=14):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
