"""Auto-generated Glaucis reference for L1 problem 27: SELU_."""

import jax
import jax.numpy as jnp


batch_size = 4096
dim = 393216


class Model:
    """
    Simple model that performs a SELU activation.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        return jax.nn.selu(x)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]


def get_init_inputs():
    return []


def simple_compute(_id=27):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
