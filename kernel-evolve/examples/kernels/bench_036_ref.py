"""Auto-generated Glaucis reference for L1 problem 36: RMSNorm_."""

import jax
import jax.numpy as jnp


batch_size = 112
features = 64
dim1 = 512
dim2 = 512


class Model:
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps

    def __call__(self, x):
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=1, keepdims=True) + self.eps)
        return x / rms


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]


def get_init_inputs():
    return [features]


def simple_compute(_id=36):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
