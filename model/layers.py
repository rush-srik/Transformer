import jax
import jax.numpy as jnp
from functools import partial

# Applies positional encoding to the input tensor
def positional_encoding(X):
    positions = jnp.arange(X.shape[-2]).reshape(-1, 1)
    dimensions = jnp.arange(X.shape[-1]).reshape(1, -1)

    angle_rates = 1 / (10000 ** ((2 * (dimensions // 2)) / (X.shape[-1])))

    pe = positions * angle_rates

    pe = jnp.where((dimensions % 2) == 0, jnp.sin(pe), jnp.cos(pe))

    return X + pe

# Computes the softmax of the input tensor
def softmax(X):
    X = X -  jnp.max(X, axis=-1, keepdims=True)
    return jnp.exp(X) / jnp.sum(jnp.exp(X), axis=-1, keepdims=True)

# Computes multi-head attention given Q, K, V, and number of heads
@partial(jax.jit, static_argnames=['num_heads', 'dropout_rate'])
def multi_head_attention(Q, K, V, num_heads, mask=None, dropout_rate=0.0, key=None):

    batch_size, seq_len, d_model = Q.shape
    d_head = d_model // num_heads

    Q_heads = Q.reshape(batch_size, seq_len, num_heads, d_head).transpose(0,2,1,3)
    K_heads = K.reshape(batch_size, seq_len, num_heads, d_head).transpose(0,2,1,3)
    V_heads = V.reshape(batch_size, seq_len, num_heads, d_head).transpose(0,2,1,3)

    scores = Q_heads @ K_heads.transpose(0,1,3,2) / jnp.sqrt(d_head)

    if mask is not None:
        scores = scores[:, None, :, :]
        scores += mask
        scores = scores[:, 0, :, :]
    
    attn = softmax(scores) @ V_heads

    if dropout_rate > 0.0:
        attn = dropout(attn, dropout_rate, key)

    return attn.transpose(0,2,1,3).reshape(batch_size, seq_len, d_model)

# Applies layer normalization to the input tensor
def layer_norm(X, gamma, beta):
    mean = jnp.mean(X, axis=-1, keepdims=True)
    variance = jnp.var(X, axis=-1, keepdims=True)

    X_norm = (X - mean) / jnp.sqrt(variance+1e-6)
    return X_norm * gamma + beta

# Computes the output of a feed-forward neural network with 3 layers and ReLU activation
def feed_forward(X, W1, b1, W2, b2):
    return jnp.maximum(0, X @ W1 + b1) @ W2 + b2

# Applies dropout to the input tensor
def dropout(X, dropout_rate, key):
    mask = jax.random.bernoulli(key, p=1-dropout_rate, shape=X.shape)
    return jnp.where(mask, X / (1 - dropout_rate), 0)