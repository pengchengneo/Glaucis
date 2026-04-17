"""Auto-generated Glaucis reference for L1 problem 13: Matmul_for_symmetric_matrices."""

import jax
import jax.numpy as jnp


N = 4096


class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.
    """
    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices.

        Args:
            A (jnp.ndarray): Input matrix A, shape (N, N), symmetric.
            B (jnp.ndarray): Input matrix B, shape (N, N), symmetric.

        Returns:
            jnp.ndarray: Output matrix C, shape (N, N).
        """
        return jnp.matmul(A, B)


def get_inputs():
    """
    Generates a pair of random symmetric matrices for testing.

    Returns:
        list: List containing two symmetric tensors A and B.
    """
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(N, N))
    A = (A + A.T) / 2  # Ensure symmetry
    B = jax.random.uniform(key2, shape=(N, N))
    B = (B + B.T) / 2  # Ensure symmetry
    return [A, B]


def get_init_inputs():
    """
    No specific initialization inputs needed for this model.

    Returns:
        list: Empty list.
    """
    return []


def simple_compute(_id=13):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
