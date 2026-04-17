"""Auto-generated Glaucis template for L1 problem 62: conv_standard_2D__square_input__asymmetric_kernel."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.
    """
    out_channels: int
    kernel_size: tuple
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        # Transpose NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        # Manual padding before conv (NHWC format)
        pad_val = self.padding if isinstance(self.padding, int) else self.padding
        if pad_val > 0:
            x = jnp.pad(x, [(0, 0), (pad_val, pad_val), (pad_val, pad_val), (0, 0)])
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=(self.stride, self.stride),
            padding='VALID',
            kernel_dilation=(self.dilation, self.dilation),
            feature_group_count=self.groups,
            use_bias=self.bias,
        )(x)
        # Transpose NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]


def get_init_inputs():
    return [out_channels, kernel_size]


# EVOLVE-BLOCK-START
def optimized_compute(_id=62):
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
