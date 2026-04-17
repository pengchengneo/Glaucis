"""Auto-generated Glaucis reference for L1 problem 98: KLDivLoss."""

import jax
import jax.numpy as jnp


batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1


class Model:
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.
    """
    def __init__(self):
        pass

    def __call__(self, predictions, targets):
        log_predictions = jnp.log(predictions)
        loss = targets * (jnp.log(targets) - log_predictions)
        return jnp.sum(loss) / predictions.shape[0]


def get_inputs():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    scale = jax.random.uniform(k1, shape=())
    return [jax.nn.softmax(jax.random.uniform(k2, shape=(batch_size, *input_shape)) * scale, axis=-1),
            jax.nn.softmax(jax.random.uniform(k3, shape=(batch_size, *input_shape)), axis=-1)]


def get_init_inputs():
    return []


def simple_compute(_id=98):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
