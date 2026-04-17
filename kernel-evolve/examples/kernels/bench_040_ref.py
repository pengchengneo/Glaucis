"""Auto-generated Glaucis reference for L1 problem 40: LayerNorm."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """
    normalized_shape: tuple

    @nn.compact
    def __call__(self, x):
        return nn.LayerNorm()(x)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]


def get_init_inputs():
    return [(features, dim1, dim2)]


def simple_compute(_id=40):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
