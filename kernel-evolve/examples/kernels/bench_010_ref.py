"""Auto-generated Glaucis reference for L1 problem 10: 3D_tensor_matrix_multiplication."""

import jax
import jax.numpy as jnp


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


def simple_compute(_id=10):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
