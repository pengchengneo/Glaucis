"""Auto-generated Glaucis template for L1 problem 64: conv_transposed_1D."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 64
in_channels = 128
out_channels = 128
kernel_size = 3
length = 65536


class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation.
    """
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    output_padding: int = 0
    groups: int = 1
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        # Transpose NCL -> NLC
        x = jnp.transpose(x, (0, 2, 1))
        x = nn.ConvTranspose(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding=[self.padding],
            use_bias=self.bias,
        )(x)
        # Transpose NLC -> NCL
        x = jnp.transpose(x, (0, 2, 1))
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, length))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]


# EVOLVE-BLOCK-START
def optimized_compute(_id=64):
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
