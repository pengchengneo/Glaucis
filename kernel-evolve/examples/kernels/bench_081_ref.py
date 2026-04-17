"""Auto-generated Glaucis reference for L1 problem 81: conv_transposed_2D_asymmetric_input_square_kernel___dilated____padded____strided__."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 64
width_in = 128
stride = 5
padding = 1
dilation = 2


class Model(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and square kernel, supporting dilation, padding, and stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel (square, e.g., 3 for a 3x3 kernel).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    bias: bool = False

    def setup(self):
        self.conv_transpose2d = nn.ConvTranspose(
            features=self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding=((self.padding, self.padding), (self.padding, self.padding)),
            kernel_dilation=(self.dilation, self.dilation),
            use_bias=self.bias,
        )

    def __call__(self, x):
        """
        Performs the 2D transposed convolution.

        Args:
            x: Input tensor of shape (batch_size, height_in, width_in, in_channels).

        Returns:
            Output tensor of shape (batch_size, height_out, width_out, out_channels).
        """
        return self.conv_transpose2d(x)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, height_in, width_in, in_channels))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]


def simple_compute(_id=81):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
