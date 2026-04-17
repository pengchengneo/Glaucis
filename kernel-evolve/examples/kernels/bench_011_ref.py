"""Auto-generated Glaucis reference for L1 problem 11: 4D_tensor_matrix_multiplication."""

import jax
import jax.numpy as jnp


b = 8
i = 256
j = 512
l = 256
k = 768


class Model:
    """
    Performs 4D tensor-matrix multiplication:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (jnp.ndarray): Input 4D tensor of shape (b, i, j, l)
        B (jnp.ndarray): Input matrix of shape (l, k)

    Returns:
        jnp.ndarray: Output 4D tensor of shape (b, i, j, k)
    """
    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication.

        Args:
            A (jnp.ndarray): Input 4D tensor of shape (b, i, j, l)
            B (jnp.ndarray): Input matrix of shape (l, k)

        Returns:
            jnp.ndarray: Output 4D tensor of shape (b, i, j, k)
        """
        return jnp.einsum("bijl,lk->bijk", A, B)


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(b, i, j, l))
    B = jax.random.uniform(key2, shape=(l, k))
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed


def simple_compute(_id=11):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
