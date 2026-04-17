"""Auto-generated Glaucis template for L1 problem 78: conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


batch_size = 8
in_channels = 32
out_channels = 32
kernel_size = (3, 7)
height = 512
width = 1024
stride = (1, 1)
padding = (1, 3)


class Model(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and kernel, with optional padding.
    """
    out_channels: int
    kernel_size: tuple
    stride: tuple = (1, 1)
    padding: tuple = (0, 0)
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = jnp.moveaxis(x, 1, -1)
        x = nn.ConvTranspose(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='VALID',
            use_bias=self.bias,
        )(x)
        # Trim output to simulate PyTorch-style ConvTranspose padding (NHWC format)
        if isinstance(self.padding, (list, tuple)):
            ph, pw = self.padding[0], self.padding[1]
        else:
            ph = pw = self.padding
        if ph > 0 or pw > 0:
            x = x[:, ph:-ph if ph > 0 else None, pw:-pw if pw > 0 else None, :]
        x = jnp.moveaxis(x, -1, 1)
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]


def get_init_inputs():
    return [out_channels, kernel_size, stride, padding]


# EVOLVE-BLOCK-START
def optimized_compute(_id=78):
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
