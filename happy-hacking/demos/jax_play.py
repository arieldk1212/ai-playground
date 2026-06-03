import jax
import jax.numpy as jnp


def predict(params, inputs):
    outputs = []
    for W, b in params:
        # y = w * i + b -> Prediction
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.tanh(outputs)  # inputs to the next layer
    return outputs  # no activation on last layer


def loss(params, inputs, targets):  # -> Loss
    preds = predict(params, inputs)
    return jnp.sum((preds - targets) ** 2)


grad_loss = jax.jit(jax.grad(loss))  # compiled gradient evaluation function
perex_grads = jax.jit(
    jax.vmap(grad_loss, in_axes=(None, 0, 0))
)  # fast per-example gradsx
