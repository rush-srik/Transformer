import jax.numpy as jnp

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

# Computes the scaled dot-product attention given Q, K, and V, optionally with a mask
def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = Q @ K.transpose(0, 2, 1) / jnp.sqrt(Q.shape[-1])
    if mask is not None:
      scores = scores[:, None, :, :]
      scores += mask
      scores = scores[:, 0, :, :]
    return softmax(scores) @ V

# Computes multi-head attention given Q, K, V, and number of heads
def multi_head_attention(Q, K, V, num_heads, mask=None):

    Q_heads = jnp.split(Q, num_heads, axis=-1)
    K_heads = jnp.split(K, num_heads, axis=-1)
    V_heads = jnp.split(V, num_heads, axis=-1)

    outputs = [scaled_dot_product_attention(Q_h, K_h, V_h, mask) for Q_h, K_h, V_h in zip(Q_heads, K_heads, V_heads)]

    return jnp.concatenate(outputs, axis=-1)

# Applies layer normalization to the input tensor
def layer_norm(X, gamma, beta):
    mean = jnp.mean(X, axis=-1, keepdims=True)
    variance = jnp.var(X, axis=-1, keepdims=True)

    X_norm = (X - mean) / jnp.sqrt(variance+1e-6)
    return X_norm * gamma + beta

# Computes the output of a feed-forward neural network with 3 layers and ReLU activation
def feed_forward(X, W1, b1, W2, b2):
    return jnp.maximum(0, X @ W1 + b1) @ W2 + b2