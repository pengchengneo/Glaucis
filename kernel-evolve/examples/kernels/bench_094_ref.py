"""Auto-generated Glaucis reference for L1 problem 94: MSELoss."""

import jax
import jax.numpy as jnp


batch_size = 32768
input_shape = (32768,)
dim = 1


class Model:
    """
    A model that computes the Mean Squared Error loss for regression tasks.
    """
    def __init__(self):
        pass

    def __call__(self, predictions, targets):
        return jnp.mean((predictions - targets) ** 2)


def get_inputs():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    scale = jax.random.uniform(k1, shape=())
    return [jax.random.uniform(k2, shape=(batch_size, *input_shape)) * scale, jax.random.uniform(k3, shape=(batch_size, *input_shape))]


def get_init_inputs():
    return []


def simple_compute(_id=94):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
