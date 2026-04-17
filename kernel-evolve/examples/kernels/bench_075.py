"""Auto-generated Glaucis template for L1 problem 75: conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (2, 3)
padding = (1, 2)
dilation = (2, 1)
groups = 4


class Model(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input, asymmetric kernel,
    grouped, padded, and dilated.
    """
    out_channels: int
    kernel_size: tuple
    stride: tuple = (1, 1)
    padding: tuple = (0, 0)
    dilation: tuple = (1, 1)
    groups: int = 1
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        # Transpose NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.ConvTranspose(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='VALID',
            kernel_dilation=self.dilation,
            use_bias=self.bias,
        )(x)
        # Trim output to simulate PyTorch-style ConvTranspose padding (NHWC format)
        if isinstance(self.padding, (list, tuple)):
            ph, pw = self.padding[0], self.padding[1]
        else:
            ph = pw = self.padding
        if ph > 0 or pw > 0:
            x = x[:, ph:-ph if ph > 0 else None, pw:-pw if pw > 0 else None, :]
        # Transpose NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]


def get_init_inputs():
    return [out_channels, kernel_size, stride, padding, dilation, groups]


# EVOLVE-BLOCK-START
def optimized_compute(_id=75):
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
