"""Auto-generated Glaucis template for L1 problem 50: conv_standard_2D__square_input__square_kernel."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 256
num_classes = 1000


class Model(nn.Module):
    """
    Standard 2D convolution with square input and square kernel.
    """
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x):
        # x is NHWC format
        x = nn.Conv(features=96, kernel_size=(11, 11), strides=(4, 4), padding=((2, 2), (2, 2)))(x)
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, 224, 224, 3))]  # NHWC


def get_init_inputs():
    return [num_classes]


# EVOLVE-BLOCK-START
def optimized_compute(_id=50):
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
