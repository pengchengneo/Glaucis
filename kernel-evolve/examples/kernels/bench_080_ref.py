"""Auto-generated Glaucis reference for L1 problem 80: conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512
stride = 1
padding = (2, 4)
dilation = (2, 3)


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.
    """
    out_channels: int
    kernel_size: tuple
    stride: int = 1
    padding: tuple = (0, 0)
    dilation: tuple = (1, 1)
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = jnp.moveaxis(x, 1, -1)
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=(self.stride, self.stride),
            padding=[self.padding[0], self.padding[1]],
            kernel_dilation=self.dilation,
            use_bias=self.bias,
        )(x)
        x = jnp.moveaxis(x, -1, 1)
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]


def get_init_inputs():
    return [out_channels, kernel_size, stride, padding, dilation]


def simple_compute(_id=80):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
