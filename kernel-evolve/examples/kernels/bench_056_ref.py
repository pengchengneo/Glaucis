"""Auto-generated Glaucis reference for L1 problem 56: conv_standard_2D__asymmetric_input__asymmetric_kernel."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 8
in_channels = 64
out_channels = 128
kernel_size = (5, 7)
height = 512
width = 256


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.
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
        # Manual padding before conv (NHWC format)
        if isinstance(self.padding, (list, tuple)):
            pad_h, pad_w = self.padding[0], self.padding[1]
        else:
            pad_h = pad_w = self.padding
        if pad_h > 0 or pad_w > 0:
            x = jnp.pad(x, [(0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)])
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='VALID',
            kernel_dilation=self.dilation,
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


def simple_compute(_id=56):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
