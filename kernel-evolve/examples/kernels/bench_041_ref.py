"""Auto-generated Glaucis reference for L1 problem 41: Max_Pooling_1D."""

import jax
import jax.numpy as jnp


batch_size = 64
features = 192
sequence_length = 65536
kernel_size = 8
stride = 1
padding = 4
dilation = 3
return_indices = False


class Model:
    """
    Simple model that performs Max Pooling 1D.
    """
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x):
        # x shape: (batch_size, features, sequence_length)
        padding_config = [(0, 0), (0, 0), (self.padding, self.padding)]
        return jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max,
            window_dimensions=(1, 1, self.kernel_size),
            window_strides=(1, 1, self.stride),
            padding=padding_config,
            window_dilation=(1, 1, self.dilation)
        )


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, sequence_length))
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]


def simple_compute(_id=41):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
