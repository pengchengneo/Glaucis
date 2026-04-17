"""Auto-generated Glaucis template for L1 problem 66: conv_standard_3D__asymmetric_input__asymmetric_kernel."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 8
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 128
width = 128


class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.
    """
    out_channels: int
    kernel_size: tuple
    stride: tuple = (1, 1, 1)
    padding: tuple = (0, 0, 0)
    dilation: tuple = (1, 1, 1)
    groups: int = 1
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        # Transpose NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        # Manual padding before conv (NDHWC format)
        if isinstance(self.padding, (list, tuple)):
            pad_d, pad_h, pad_w = self.padding[0], self.padding[1], self.padding[2]
        else:
            pad_d = pad_h = pad_w = self.padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = jnp.pad(x, [(0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w), (0, 0)])
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='VALID',
            kernel_dilation=self.dilation,
            feature_group_count=self.groups,
            use_bias=self.bias,
        )(x)
        # Transpose NDHWC -> NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, depth, height, width))
    return [x]


def get_init_inputs():
    return [out_channels, kernel_size]


# EVOLVE-BLOCK-START
def optimized_compute(_id=66):
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
