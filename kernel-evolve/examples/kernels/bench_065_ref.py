"""Auto-generated Glaucis reference for L1 problem 65: conv_transposed_2D__square_input__asymmetric_kernel."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 8
in_channels = 64
out_channels = 64
kernel_size = (3, 7)
width = 512
height = 512


class Model(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.
    """
    out_channels: int
    kernel_size: tuple
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
            kernel_size=self.kernel_size,
            strides=(self.stride, self.stride),
            padding='VALID',
            use_bias=self.bias,
        )(x)
        # Trim output to simulate PyTorch-style ConvTranspose padding (NHWC format)
        pad_val = self.padding if isinstance(self.padding, int) else self.padding
        if pad_val > 0:
            x = x[:, pad_val:-pad_val, pad_val:-pad_val, :]
        # Transpose NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]


def get_init_inputs():
    return [out_channels, kernel_size]


def simple_compute(_id=65):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
