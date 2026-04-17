"""Auto-generated Glaucis reference for L1 problem 70: conv_transposed_3D__asymmetric_input__square_kernel."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 2
in_channels = 48
out_channels = 24
kernel_size = 3
depth = 96
height = 96
width = 96


class Model(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and a square kernel.
    """
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    output_padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        # Transpose NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = nn.ConvTranspose(
            features=self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride, self.stride),
            padding=[self.padding, self.padding, self.padding],
            kernel_dilation=(self.dilation, self.dilation, self.dilation),
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


def simple_compute(_id=70):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
