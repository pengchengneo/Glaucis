"""Auto-generated Glaucis reference for L1 problem 20: LeakyReLU."""

import jax
import jax.numpy as jnp


batch_size = 4096
dim = 393216


class Model:
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        self.negative_slope = negative_slope

    def __call__(self, x):
        """
        Applies LeakyReLU activation to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor of any shape.

        Returns:
            jnp.ndarray: Output tensor with LeakyReLU applied, same shape as input.
        """
        return jax.nn.leaky_relu(x, negative_slope=self.negative_slope)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed


def simple_compute(_id=20):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
