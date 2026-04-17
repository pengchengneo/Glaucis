"""Auto-generated Glaucis reference for L1 problem 60: conv_standard_3D__square_input__asymmetric_kernel."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)
width = 64
height = 64
depth = 64


class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.
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
        # Transpose NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        # Manual padding before conv (NDHWC format)
        pad_val = self.padding if isinstance(self.padding, int) else self.padding
        if pad_val > 0:
            x = jnp.pad(x, [(0, 0), (pad_val, pad_val), (pad_val, pad_val), (pad_val, pad_val), (0, 0)])
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=(self.stride, self.stride, self.stride),
            padding='VALID',
            kernel_dilation=(self.dilation, self.dilation, self.dilation),
            feature_group_count=self.groups,
            use_bias=self.bias,
        )(x)
        # Transpose NDHWC -> NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, width, height, depth))
    return [x]


def get_init_inputs():
    return [out_channels, kernel_size]


def simple_compute(_id=60):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
