"""Auto-generated Glaucis reference for L1 problem 84: conv_depthwise_2D_asymmetric_input_square_kernel."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 64
in_channels = 128
out_channels = 128
kernel_size = 3
width_in = 512
height_in = 256
stride = 1
padding = 0


class Model(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    bias: bool = False

    def setup(self):
        self.conv2d = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding=((self.padding, self.padding), (self.padding, self.padding)),
            feature_group_count=self.in_channels,
            use_bias=self.bias,
        )

    def __call__(self, x):
        """
        Performs the depthwise 2D convolution.

        Args:
            x: Input tensor of shape (batch_size, height_in, width_in, in_channels).

        Returns:
            Output tensor of shape (batch_size, height_out, width_out, out_channels).
        """
        return self.conv2d(x)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, height_in, width_in, in_channels))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]


def simple_compute(_id=84):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
