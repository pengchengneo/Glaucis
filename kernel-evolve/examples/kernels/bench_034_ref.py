"""Auto-generated Glaucis reference for L1 problem 34: InstanceNorm."""

import jax
import jax.numpy as jnp


batch_size = 112
features = 64
dim1 = 512
dim2 = 512


class Model:
    """
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps

    def __call__(self, x):
        # x shape: (batch_size, num_features, height, width)
        mean = jnp.mean(x, axis=(2, 3), keepdims=True)
        var = jnp.var(x, axis=(2, 3), keepdims=True)
        return (x - mean) / jnp.sqrt(var + self.eps)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]


def get_init_inputs():
    return [features]


def simple_compute(_id=34):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
