import jax
import jax.numpy as jnp
from jax import random
from model.blocks import encoder_decoder_transformer, decoder_only_transformer
from functools import partial
import time

# Computes the cross entropy loss for a sample
def cross_entropy_loss(parameters, n_heads, y, encdec=True, x=None, dropout_rate=0.0, key=None):
  tgt_in = jnp.concatenate((jnp.zeros((y.shape[0], 1), dtype=y.dtype), y[:, :-1]), axis=1)
  if encdec:
      logits = encoder_decoder_transformer(x, tgt_in, n_heads, parameters)
  else:
      logits = decoder_only_transformer(tgt_in, n_heads, parameters, dropout_rate=dropout_rate, key=key)

  logp = jax.nn.log_softmax(logits, axis=-1)
  likelihoods = -jnp.take_along_axis(logp, jnp.expand_dims(y, -1), axis=-1).squeeze(-1)
  return jnp.mean(likelihoods)

# Initializes the Adam optimizer state
def init_adam_state(parameters):
    m = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    v = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    t = 0
    return m, v, t

# Performs an Adam update step
@jax.jit
def adam_update(parameters, grads, m, v, t, lr, weight_decay, b1=0.9, b2=0.999, eps=1e-8):
    t += 1
    
    m = jax.tree_util.tree_map(lambda m_next, gd: b1 * m_next + (1 - b1) * gd, m, grads)
    
    v = jax.tree_util.tree_map(lambda v_next, gd: b2 * v_next + (1 - b2) * (gd**2), v, grads)
    
    m_hat = jax.tree_util.tree_map(lambda m_next: m_next / (1 - b1**t), m)
    v_hat = jax.tree_util.tree_map(lambda v_next: v_next / (1 - b2**t), v)

    parameters = jax.tree_util.tree_map(
        lambda p, mh, vh: p * (1 - lr * weight_decay) - lr * mh / (jnp.sqrt(vh) + eps), 
        parameters, m_hat, v_hat
    )

    return parameters, m, v, t

# Performs a training step with Adam
@partial(jax.jit, static_argnames=['n_heads', 'dropout_rate'])
def adam_step(parameters, n_heads, y, lr, m, v, t, weight_decay, dropout_rate=0.0, key=None):
   loss, grads = jax.value_and_grad(cross_entropy_loss)(parameters, n_heads, y, encdec=False, dropout_rate=dropout_rate, key=key)
   parameters, m, v, t = adam_update(parameters, grads, m, v, t, lr, weight_decay)

   return loss, parameters, m, v, t

# Implements a learning rate scheduler with warmup and cosine decay
def get_lr(step, total_steps, base_lr, min_lr, warmup_steps):
   if step < warmup_steps:
      return base_lr * (step+1) / warmup_steps
   
   decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
   coeff = 0.5 * (1 + jnp.cos(jnp.pi * decay_ratio))
    
   return min_lr + coeff * (base_lr - min_lr)
   
# Performs a training step, returning the loss and updated parameters
@partial(jax.jit, static_argnames=['n_heads', 'encdec'])
def train_step(parameters, n_heads, y, lr, encdec=True, x=None):
  loss, grads = jax.value_and_grad(cross_entropy_loss)(parameters, n_heads, y, encdec=encdec, x=x)
  parameters = jax.tree_util.tree_map(lambda p, g: p - lr * g, parameters, grads)
  return loss, parameters

# Batches inputs X and y
def batcher(X, y, batch_size, key):

  n = X.shape[0]

  key, permute_key = random.split(key)
  shuffle_idx = random.permutation(permute_key, n)

  X, y = X[shuffle_idx], y[shuffle_idx]

  num_batches = n // batch_size
  X = X[:num_batches*batch_size]
  y = y[:num_batches*batch_size]

  X = X.reshape((num_batches, batch_size, X.shape[1]))
  y = y.reshape((num_batches, batch_size, y.shape[1]))

  return X, y

# Splits data into training and testing sets
def train_test_split(X, y, test_size, key):
  X, y = jax.random.permutation(key, X), jax.random.permutation(key, y)

  n = jnp.ceil(X.shape[0] * (1-test_size)).astype(int)
  X_train, y_train = X[:n], y[:n]
  X_test, y_test = X[n:], y[n:]

  return X_train, X_test, y_train, y_test

# Performs inference on a source sequence using an encoder-decoder transformer
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

# One step of generation using the decoder
@partial(jax.jit, static_argnames=['n_heads', 'temperature'])
def generate_step(parameters, n_heads, temperature, current_context, key):
    
    logits = decoder_only_transformer(current_context, n_heads, parameters)
    next_token_logits = logits[:, -1, :]
    
    next_token_logits = next_token_logits.at[:, 0].set(-1e9)
    next_token_logits /= temperature
    
    key, gen_key = random.split(key)
    next_token = random.categorical(gen_key, next_token_logits, axis=-1)
    
    # Sliding window
    new_context = jnp.concatenate([current_context[:, 1:], next_token[:, None]], axis=1)
    
    return new_context, next_token, key

# Performs inference using the decoder
def predict_dec_only(parameters, n_heads, block_size, max_length, temperature, start_tokens, key):

    L = len(start_tokens)
    buffer = jnp.zeros((1, block_size), dtype=jnp.int32)
    buffer = buffer.at[:, -L:].set(jnp.array(start_tokens))

    generated = []
    
    for _ in range(max_length):
        buffer, next_token, key = generate_step(parameters, n_heads, temperature, buffer, key)
        
        generated.append(int(next_token[0]))

    return start_tokens + generated

# Computes accuracy
def accuracy(y_true, y_pred):
    return jnp.mean(jnp.all(y_true == y_pred, axis=1))