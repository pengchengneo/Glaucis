"""Auto-generated Glaucis template for L1 problem 35: GroupNorm_."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 112
features = 64
num_groups = 8
dim1 = 512
dim2 = 512


class Model(nn.Module):
    """
    Simple model that performs Group Normalization.
    """
    num_features: int
    num_groups: int

    @nn.compact
    def __call__(self, x):
        return nn.GroupNorm(num_groups=self.num_groups)(x)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim1, dim2, features))  # NHWC
    return [x]


def get_init_inputs():
    return [features, num_groups]


# EVOLVE-BLOCK-START
def optimized_compute(_id=35):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)
# EVOLVE-BLOCK-END
