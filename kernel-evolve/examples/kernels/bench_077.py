"""Auto-generated Glaucis template for L1 problem 77: conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 1
dilation = 2


class Model(nn.Module):
    """
    Performs a 3D transposed convolution operation with square input and square kernel,
    and supports padding, dilation, and stride.
    """
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = jnp.moveaxis(x, 1, -1)
        x = nn.ConvTranspose(
            features=self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride, self.stride),
            padding=((self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)),
            kernel_dilation=(self.dilation, self.dilation, self.dilation),
            use_bias=self.bias,
        )(x)
        x = jnp.moveaxis(x, -1, 1)
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, depth, height, width))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]


# EVOLVE-BLOCK-START
def optimized_compute(_id=77):
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
