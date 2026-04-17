"""Auto-generated Glaucis reference for L1 problem 44: Average_Pooling_1D."""

import jax
import jax.numpy as jnp


batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4


class Model:
    """
    Simple model that performs 1D Average Pooling.
    """
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x):
        # x shape: (batch_size, in_channels, input_length)
        padding_config = [(0, 0), (0, 0), (self.padding, self.padding)]
        sums = jax.lax.reduce_window(
            x, 0.0, jax.lax.add,
            window_dimensions=(1, 1, self.kernel_size),
            window_strides=(1, 1, self.stride),
            padding=padding_config
        )
        return sums / self.kernel_size


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, input_length))
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding]


def simple_compute(_id=44):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
