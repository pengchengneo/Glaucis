"""Auto-generated Glaucis reference for L1 problem 19: ReLU."""

import jax
import jax.numpy as jnp


batch_size = 4096
dim = 393216


class Model:
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor of any shape.

        Returns:
            jnp.ndarray: Output tensor with ReLU applied, same shape as input.
        """
        return jax.nn.relu(x)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed


def simple_compute(_id=19):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
