"""Auto-generated Glaucis template for L1 problem 74: conv_transposed_1D_dilated."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 32
in_channels = 32
out_channels = 64
kernel_size = 5
length = 131072
stride = 1
padding = 0
dilation = 3


class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.
    """
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1
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
            kernel_dilation=(self.dilation,),
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
    return [out_channels, kernel_size, stride, padding, dilation]


# EVOLVE-BLOCK-START
def optimized_compute(_id=74):
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
