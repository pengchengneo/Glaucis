"""Auto-generated Glaucis template for L1 problem 87: conv_pointwise_2D."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024


class Model(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    in_channels: int
    out_channels: int
    bias: bool = False

    def setup(self):
        self.conv1d = nn.Conv(
            features=self.out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=((0, 0), (0, 0)),
            use_bias=self.bias,
        )

    def __call__(self, x):
        """
        Performs the pointwise 2D convolution.

        Args:
            x: Input tensor of shape (batch_size, height, width, in_channels).

        Returns:
            Output tensor of shape (batch_size, height, width, out_channels).
        """
        return self.conv1d(x)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, height, width, in_channels))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels]


# EVOLVE-BLOCK-START
def optimized_compute(_id=87):
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
