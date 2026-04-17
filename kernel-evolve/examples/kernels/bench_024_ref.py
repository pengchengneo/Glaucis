"""Auto-generated Glaucis reference for L1 problem 24: LogSoftmax."""

import jax
import jax.numpy as jnp


batch_size = 4096
dim = 393216


class Model:
    """
    Simple model that performs a LogSoftmax activation.
    """
    def __init__(self, dim: int = 1):
        self.dim = dim

    def __call__(self, x):
        """
        Applies LogSoftmax activation to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, dim).

        Returns:
            jnp.ndarray: Output tensor with LogSoftmax applied, same shape as input.
        """
        return jax.nn.log_softmax(x, axis=self.dim)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed


def simple_compute(_id=24):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
