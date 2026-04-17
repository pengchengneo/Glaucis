"""Auto-generated Glaucis template for L1 problem 72: conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 8
in_channels = 32
out_channels = 32
kernel_size = (3, 5, 7)
depth = 12
height = 24
width = 48
stride = (2, 2, 2)
padding = (1, 2, 3)
output_padding = (1, 1, 1)
groups = 4


class Model(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and kernel, and optional stride.
    """
    out_channels: int
    kernel_size: tuple
    stride: tuple = (1, 1, 1)
    padding: tuple = (0, 0, 0)
    output_padding: tuple = (0, 0, 0)
    groups: int = 1
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        # Transpose NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = nn.ConvTranspose(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='VALID',
            use_bias=self.bias,
        )(x)
        # Trim output to simulate PyTorch-style ConvTranspose padding (NDHWC format)
        if isinstance(self.padding, (list, tuple)):
            pd, ph, pw = self.padding[0], self.padding[1], self.padding[2]
        else:
            pd = ph = pw = self.padding
        if pd > 0 or ph > 0 or pw > 0:
            x = x[:, pd:-pd if pd > 0 else None, ph:-ph if ph > 0 else None, pw:-pw if pw > 0 else None, :]
        # Transpose NDHWC -> NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, depth, height, width))
    return [x]


def get_init_inputs():
    return [out_channels, kernel_size, stride, padding, output_padding, groups]


# EVOLVE-BLOCK-START
def optimized_compute(_id=72):
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
