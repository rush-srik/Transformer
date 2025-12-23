import jax
import jax.numpy as jnp
from jax import random
from model.blocks import encoder_decoder_transformer
from functools import partial

# Computes the cross entropy loss for a sample
def cross_entropy_loss(parameters, n_heads, x, y):
    tgt_in = jnp.concatenate((jnp.zeros((y.shape[0], 1), dtype=y.dtype), y[:, :-1]), axis=1)
    logits = encoder_decoder_transformer(x, tgt_in, n_heads, parameters)

    logp = jax.nn.log_softmax(logits, axis=-1)
    likelihoods = -jnp.take_along_axis(logp, jnp.expand_dims(y, -1), axis=-1).squeeze(-1)
    return jnp.mean(likelihoods)

grad_loss = jax.grad(cross_entropy_loss)

# Performs a training step, returning the loss and updated parameters
@partial(jax.jit, static_argnames=['n_heads'])
def train_step(parameters, n_heads, x, y, lr):
  loss = cross_entropy_loss(parameters, n_heads, x, y)
  grads = grad_loss(parameters, n_heads, x, y)
  parameters = jax.tree_util.tree_map(lambda p, g: p - lr * g, parameters, grads)
  return loss, parameters

# Batches inputs X and y
def batcher(X, y, batch_size, key):

  n = X.shape[0]

  key, permute_key = random.split(key)
  shuffle_idx = random.permutation(permute_key, n)

  X, y = X[shuffle_idx], y[shuffle_idx]

  n_batches = jnp.ceil(n / batch_size).astype(int)

  batches = []
  for i in range(n_batches):
    start = batch_size*i
    end = batch_size*(i+1)
    batches.append((X[start:end], y[start:end]))

  return batches

# Splits data into training and testing sets
def train_test_split(X, y, test_size, key):
  X, y = jax.random.permutation(key, X), jax.random.permutation(key, y)

  n = jnp.ceil(X.shape[0] * (1-test_size)).astype(int)
  X_train, y_train = X[:n], y[:n]
  X_test, y_test = X[n:], y[n:]

  return X_train, X_test, y_train, y_test

# Performs inference on a source sequence
def predict(src, params, num_heads):
    if src.ndim == 1:
        src = src[None, :]

    batch, length = src.shape

    tgt = jnp.zeros((batch, length + 1), dtype=src.dtype)

    def step(i, tgt):
        mask = jnp.where((jnp.arange(length + 1) <= i)[None, None, :], 0.0, -1e9)

        logits = encoder_decoder_transformer(src, tgt, num_heads, params, mask=mask)

        next_token = jnp.argmax(logits[:, i, :], axis=-1)

        tgt = tgt.at[:, i+1].set(next_token)
        
        return tgt

    tgt = jax.lax.fori_loop(0, length, step, tgt)

    return tgt[:, 1:]

# Computes accuracy
def accuracy(y_true, y_pred):
    return jnp.mean(jnp.all(y_true == y_pred, axis=1))