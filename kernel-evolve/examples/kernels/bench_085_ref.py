"""Auto-generated Glaucis reference for L1 problem 85: conv_depthwise_2D_asymmetric_input_asymmetric_kernel."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 32
in_channels = 128
out_channels = 128
kernel_size_h = 3
kernel_size_w = 7
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels


class Model(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size_h (int): Height of the convolution kernel.
        kernel_size_w (int): Width of the convolution kernel.
        stride_h (int, optional): Stride of the convolution in height dimension. Defaults to 1.
        stride_w (int, optional): Stride of the convolution in width dimension. Defaults to 1.
        padding_h (int, optional): Padding applied to the input in height dimension. Defaults to 0.
        padding_w (int, optional): Padding applied to the input in width dimension. Defaults to 0.
        dilation_h (int, optional): Spacing between kernel elements in height dimension. Defaults to 1.
        dilation_w (int, optional): Spacing between kernel elements in width dimension. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    in_channels: int
    out_channels: int
    kernel_size_h: int
    kernel_size_w: int
    stride_h: int = 1
    stride_w: int = 1
    padding_h: int = 0
    padding_w: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    groups: int = 1
    bias: bool = False

    def setup(self):
        self.conv2d = nn.Conv(
            features=self.in_channels,
            kernel_size=(self.kernel_size_h, self.kernel_size_w),
            strides=(self.stride_h, self.stride_w),
            padding=((self.padding_h, self.padding_h), (self.padding_w, self.padding_w)),
            kernel_dilation=(self.dilation_h, self.dilation_w),
            feature_group_count=self.in_channels,
            use_bias=self.bias,
        )

    def __call__(self, x):
        """
        Performs the depthwise 2D convolution.

        Args:
            x: Input tensor of shape (batch_size, height, width, in_channels).

        Returns:
            Output tensor of shape (batch_size, height_out, width_out, out_channels).
        """
        return self.conv2d(x)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, height, width, in_channels))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]


def simple_compute(_id=85):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
