"""Auto-generated Glaucis reference for L1 problem 42: Max_Pooling_2D."""

import jax
import jax.numpy as jnp


batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1


class Model:
    """
    Simple model that performs Max Pooling 2D.
    """
    def __init__(self, kernel_size, stride, padding, dilation):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x):
        # x shape: (batch_size, channels, height, width)
        padding_config = [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)]
        return jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max,
            window_dimensions=(1, 1, self.kernel_size, self.kernel_size),
            window_strides=(1, 1, self.stride, self.stride),
            padding=padding_config,
            window_dilation=(1, 1, self.dilation, self.dilation)
        )


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, channels, height, width))
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]


def simple_compute(_id=42):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
