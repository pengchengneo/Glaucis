"""Auto-generated Glaucis reference for L1 problem 79: conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__."""

import jax
import jax.numpy as jnp
from flax import linen as nn


batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
length = 131072
stride = 2
padding = 1
dilation = 2


class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation with asymmetric input and square kernel.
    Supports padding, striding, and dilation.
    """
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = jnp.moveaxis(x, 1, -1)
        x = nn.ConvTranspose(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding=((self.padding, self.padding),),
            kernel_dilation=(self.dilation,),
            use_bias=self.bias,
        )(x)
        x = jnp.moveaxis(x, -1, 1)
        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, length))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]


def simple_compute(_id=79):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    params = model.init(jax.random.PRNGKey(42), *inputs)
    return model.apply(params, *inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
