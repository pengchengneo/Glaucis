"""Auto-generated Glaucis reference for L1 problem 92: cumsum_exclusive."""

import jax
import jax.numpy as jnp


batch_size = 32768
input_shape = (32768,)
dim = 1


class Model:
    """
    A model that performs an exclusive cumulative sum (does not include the current element).
    """
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        cumsum = jnp.cumsum(
            jax.lax.slice_in_dim(x, 0, x.shape[self.dim] - 1, axis=self.dim),
            axis=self.dim,
        )
        zeros_shape = list(x.shape)
        zeros_shape[self.dim] = 1
        return jnp.concatenate([jnp.zeros(zeros_shape), cumsum], axis=self.dim)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]


def get_init_inputs():
    return [dim]


def simple_compute(_id=92):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
