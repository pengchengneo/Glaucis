"""Auto-generated Glaucis reference for L1 problem 95: CrossEntropyLoss."""

import jax
import jax.numpy as jnp


batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1


class Model:
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.
    """
    def __init__(self):
        pass

    def __call__(self, predictions, targets):
        log_probs = jax.nn.log_softmax(predictions, axis=-1)
        nll = -log_probs[jnp.arange(predictions.shape[0]), targets]
        return jnp.mean(nll)


def get_inputs():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    return [jax.random.uniform(k1, shape=(batch_size, *input_shape)), jax.random.randint(k2, shape=(batch_size,), minval=0, maxval=num_classes)]


def get_init_inputs():
    return []


def simple_compute(_id=95):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
