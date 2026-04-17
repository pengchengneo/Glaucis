"""Auto-generated Glaucis template for L1 problem 82: conv_depthwise_2D_square_input_square_kernel."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0


class Model(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    in_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    bias: bool = False

    def setup(self):
        self.conv2d = nn.Conv(
            features=self.in_channels,
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
            x: Input tensor of shape (batch_size, height, width, in_channels).

        Returns:
            Output tensor of shape (batch_size, height_out, width_out, in_channels).
        """
        return self.conv2d(x)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, height, width, in_channels))
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]


# EVOLVE-BLOCK-START
def optimized_compute(_id=82):
    """Replace this with a Pallas kernel implementation.

    The original Model class and get_inputs/get_init_inputs are available
    at module scope above for reference. Your optimized version should use
    jax.experimental.pallas.pallas_call for the core computation.
    """
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)
# EVOLVE-BLOCK-END
