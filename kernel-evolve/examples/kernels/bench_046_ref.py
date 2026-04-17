"""Auto-generated Glaucis reference for L1 problem 46: Average_Pooling_3D."""

import jax
import jax.numpy as jnp


batch_size = 16
channels = 32
depth = 128
height = 128
width = 256
kernel_size = 3
stride = 2
padding = 1


class Model:
    """
    Simple model that performs 3D Average Pooling.
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def __call__(self, x):
        # x shape: (batch_size, channels, depth, height, width)
        padding_config = [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)]
        sums = jax.lax.reduce_window(
            x, 0.0, jax.lax.add,
            window_dimensions=(1, 1, self.kernel_size, self.kernel_size, self.kernel_size),
            window_strides=(1, 1, self.stride, self.stride, self.stride),
            padding=padding_config
        )
        return sums / (self.kernel_size ** 3)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, channels, depth, height, width))
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding]


def simple_compute(_id=46):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
