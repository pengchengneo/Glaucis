"""Auto-generated Glaucis reference for L1 problem 37: FrobeniusNorm_."""

import jax
import jax.numpy as jnp


batch_size = 112
features = 64
dim1 = 512
dim2 = 512


class Model:
    """
    Simple model that performs Frobenius norm normalization.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        norm = jnp.sqrt(jnp.sum(x ** 2))
        return x / norm


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]


def get_init_inputs():
    return []


def simple_compute(_id=37):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
