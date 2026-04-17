"""Auto-generated Glaucis template for L1 problem 76: conv_standard_1D_dilated_strided__."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 64
in_channels = 64
out_channels = 128
kernel_size = 3
length = 524280
stride = 3
dilation = 4


class Model(nn.Module):
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, potentially dilated and strided.
    """
    out_channels: int
    kernel_size: int
    stride: int = 1
    dilation: int = 1
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = jnp.moveaxis(x, 1, -1)
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding='VALID',
            kernel_dilation=(self.dilation,),
            use_bias=self.bias,
        )(x)
        x = jnp.moveaxis(x, -1, 1)
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, length))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation]


# EVOLVE-BLOCK-START
def optimized_compute(_id=76):
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
