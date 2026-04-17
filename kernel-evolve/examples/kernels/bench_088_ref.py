"""Auto-generated Glaucis reference for L1 problem 88: MinGPTNewGelu."""

import jax
import jax.numpy as jnp


batch_size = 8192
dim = 8192


class Model:
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        pass

    def __call__(self, x):
        return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, dim))]


def get_init_inputs():
    return []


def simple_compute(_id=88):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
