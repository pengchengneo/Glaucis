"""Auto-generated Glaucis reference for L1 problem 38: L1Norm_."""

import jax
import jax.numpy as jnp


batch_size = 32768
dim = 65535


class Model:
    """
    Simple model that performs L1 normalization.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        return x / jnp.mean(jnp.abs(x), axis=1, keepdims=True)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]


def get_init_inputs():
    return []


def simple_compute(_id=38):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
