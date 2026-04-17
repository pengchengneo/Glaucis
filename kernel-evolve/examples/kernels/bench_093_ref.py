"""Auto-generated Glaucis reference for L1 problem 93: masked_cumsum."""

import jax
import jax.numpy as jnp


batch_size = 32768
input_shape = (32768,)
dim = 1


class Model:
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.
    """
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x, mask):
        return jnp.cumsum(x * mask, axis=self.dim)


def get_inputs():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1, shape=(batch_size, *input_shape))
    mask = jax.random.randint(k2, shape=x.shape, minval=0, maxval=2).astype(jnp.bool_)
    return [x, mask]


def get_init_inputs():
    return [dim]


def simple_compute(_id=93):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
