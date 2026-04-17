"""Auto-generated Glaucis reference for L1 problem 89: cumsum."""

import jax
import jax.numpy as jnp


batch_size = 32768
input_shape = (32768,)
dim = 1


class Model:
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        self.dim = dim

    def __call__(self, x):
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x: Input tensor of shape (batch_size, *input_shape), where `*input_shape`
               can vary depending on the use case.

        Returns:
            Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        return jnp.cumsum(x, axis=self.dim)


def get_inputs():
    """
    Generates random inputs for testing the Scan model.

    Returns:
        list: A list containing a single randomly generated tensor with shape
              (batch_size, *input_shape).
    """
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]


def get_init_inputs():
    """
    Returns the initialization parameters for the Scan model.

    Returns:
        list: A list containing the `dim` parameter for model initialization.
    """
    return [dim]


def simple_compute(_id=89):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
