"""Auto-generated Glaucis reference for L1 problem 73: conv_transposed_3D_asymmetric_input_square_kernel__strided_padded__grouped."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 4
in_channels = 32
out_channels = 32
kernel_size = 3
depth = 32
height = 64
width = 128
stride = 2
padding = 1
groups = 4


class Model(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and square kernel.
    The input is padded before the convolution.
    """
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
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
    return [out_channels, kernel_size, stride, padding, groups]


def simple_compute(_id=73):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
