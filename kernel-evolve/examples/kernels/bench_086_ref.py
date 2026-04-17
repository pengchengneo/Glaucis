"""Auto-generated Glaucis reference for L1 problem 86: conv_depthwise_separable_2D."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 16
in_channels = 64
out_channels = 128
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 1
dilation = 1


class Model(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
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
        self.depthwise = nn.Conv(
            features=self.in_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding=((self.padding, self.padding), (self.padding, self.padding)),
            kernel_dilation=(self.dilation, self.dilation),
            feature_group_count=self.in_channels,
            use_bias=self.bias,
        )
        self.pointwise = nn.Conv(
            features=self.out_channels,
            kernel_size=(1, 1),
            use_bias=self.bias,
        )

    def __call__(self, x):
        """
        Performs the depthwise-separable 2D convolution.

        Args:
            x: Input tensor of shape (batch_size, height, width, in_channels).

        Returns:
            Output tensor of shape (batch_size, height_out, width_out, out_channels).
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, height, width, in_channels))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]


def simple_compute(_id=86):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
