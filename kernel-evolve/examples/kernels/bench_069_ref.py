"""Auto-generated Glaucis reference for L1 problem 69: conv_transposed_2D__asymmetric_input__asymmetric_kernel."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 64
in_channels = 64
out_channels = 128
kernel_size = (3, 5)
height_in = 128
width_in = 256


class Model(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.
    """
    out_channels: int
    kernel_size: tuple
    stride: tuple = (1, 1)
    padding: tuple = (0, 0)
    output_padding: tuple = (0, 0)
    dilation: tuple = (1, 1)
    groups: int = 1
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        # Transpose NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.ConvTranspose(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='VALID',
            kernel_dilation=self.dilation,
            use_bias=self.bias,
        )(x)
        # Trim output to simulate PyTorch-style ConvTranspose padding (NHWC format)
        if isinstance(self.padding, (list, tuple)):
            ph, pw = self.padding[0], self.padding[1]
        else:
            ph = pw = self.padding
        if ph > 0 or pw > 0:
            x = x[:, ph:-ph if ph > 0 else None, pw:-pw if pw > 0 else None, :]
        # Transpose NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height_in, width_in))
    return [x]


def get_init_inputs():
    return [out_channels, kernel_size]


def simple_compute(_id=69):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
