"""Auto-generated Glaucis reference for L1 problem 57: conv_transposed_2D__square_input__square_kernel."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 8
in_channels = 64
out_channels = 64
kernel_size = 3
height = 1024
width = 1024


class Model(nn.Module):
    """
    Performs a transposed 2D convolution with square input and square kernel.
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
        # Transpose NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.ConvTranspose(
            features=self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding=[self.padding, self.padding],
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
    return [in_channels, out_channels, kernel_size]


def simple_compute(_id=57):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
