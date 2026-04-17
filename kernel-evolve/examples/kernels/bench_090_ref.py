"""Auto-generated Glaucis reference for L1 problem 90: cumprod."""

import jax
import jax.numpy as jnp


batch_size = 32768
input_shape = (32768,)
dim = 1


class Model:
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        self.dim = dim

    def __call__(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x: Input tensor of shape (batch_size, *input_shape).

        Returns:
            Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return jnp.cumprod(x, axis=self.dim)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]


def get_init_inputs():
    return [dim]


def simple_compute(_id=90):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
