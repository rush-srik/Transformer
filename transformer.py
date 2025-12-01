import jax
import jax.numpy as jnp
from jax import random

seq_len = 64
d_model = 128
num_heads = 8
n_hidden = 4 * d_model
n_layers = 2
n_vocab = 1000

n_samples = 256
batch_size = 16
epochs = 500
lr = 1e-1

key = random.key(42)

# Applies positional encoding to the input tensor
def positional_encoding(X):
    positions = jnp.arange(X.shape[-2]).reshape(-1, 1)
    dimensions = jnp.arange(X.shape[-1]).reshape(1, -1)

    angle_rates = 1 / (10000 ** ((2 * (dimensions // 2)) / (d_model)))

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
      scores = scores.squeeze()
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

# Initializes the embedding matrix
def init_embedding(key):
    return random.normal(key, (n_vocab, d_model)) * (1.0 / jnp.sqrt(d_model))

# Initializes the parameters for an encoder block
def init_encoder(key):

    keys = random.split(key, 8)
    parameters = {}

    parameters['W_Q'] = random.normal(keys[0], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    parameters['W_K'] = random.normal(keys[1], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    parameters['W_V'] = random.normal(keys[2], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    parameters['W_O'] = random.normal(keys[3], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))

    parameters['gamma1'] = jnp.ones(d_model)
    parameters['beta1'] = jnp.zeros(d_model)

    parameters['W1'] = random.normal(keys[4], (d_model, n_hidden)) * (1.0 / jnp.sqrt(d_model))
    parameters['b1'] = random.normal(keys[5], (n_hidden,)) * (1.0 / jnp.sqrt(d_model))
    parameters['W2'] = random.normal(keys[6], (n_hidden, d_model)) * (1.0 / jnp.sqrt(d_model))
    parameters['b2'] = random.normal(keys[7], (d_model,)) * (1.0 / jnp.sqrt(d_model))

    parameters['gamma2'] = jnp.ones(d_model)
    parameters['beta2'] = jnp.zeros(d_model)

    return parameters

# Implements a single encoder block
def encoder_block(X, parameters):

    Q = X @ parameters['W_Q']
    K = X @ parameters['W_K']
    V = X @ parameters['W_V']

    Z = multi_head_attention(Q, K, V, num_heads)
    Z = Z @ parameters['W_O']

    X = layer_norm(X + Z, parameters['gamma1'], parameters['beta1'])

    ff_out = feed_forward(X, parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2'])

    return layer_norm(X + ff_out, parameters['gamma2'], parameters['beta2'])

# Initializes the parameters for a decoder block
def init_decoder(key):

    keys = random.split(key, 12)
    parameters = {}

    parameters['W_Q'] = random.normal(keys[0], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    parameters['W_K'] = random.normal(keys[1], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    parameters['W_V'] = random.normal(keys[2], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    parameters['W_O'] = random.normal(keys[3], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))

    parameters['gamma1'] = jnp.ones(d_model)
    parameters['beta1'] = jnp.zeros(d_model)

    parameters['W_Q2'] = random.normal(keys[4], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    parameters['W_K2'] = random.normal(keys[5], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    parameters['W_V2'] = random.normal(keys[6], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    parameters['W_O2'] = random.normal(keys[7], (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))

    parameters['gamma2'] = jnp.ones(d_model)
    parameters['beta2'] = jnp.zeros(d_model)

    parameters['W1'] = random.normal(keys[8], (d_model, n_hidden)) * (1.0 / jnp.sqrt(d_model))
    parameters['b1'] = random.normal(keys[9], (n_hidden,)) * (1.0 / jnp.sqrt(d_model))
    parameters['W2'] = random.normal(keys[10], (n_hidden, d_model)) * (1.0 / jnp.sqrt(d_model))
    parameters['b2'] = random.normal(keys[11], (d_model,)) * (1.0 / jnp.sqrt(d_model))

    parameters['gamma3'] = jnp.ones(d_model)
    parameters['beta3'] = jnp.zeros(d_model)

    return parameters

# Implements a single decoder block
def decoder_block(X, enc_output, parameters):

    Q = X @ parameters['W_Q']
    K = X @ parameters['W_K']
    V = X @ parameters['W_V']

    mask = jnp.triu(jnp.full((1, 1, X.shape[1], X.shape[1]), -1e9), k=1)

    Z = multi_head_attention(Q, K, V, num_heads, mask=mask)
    Z = Z @ parameters['W_O']

    X = layer_norm(X + Z, parameters['gamma1'], parameters['beta1'])

    Q2 = X @ parameters['W_Q2']
    K2 = enc_output @ parameters['W_K2']
    V2 = enc_output @ parameters['W_V2']

    Z2 = multi_head_attention(Q2, K2, V2, num_heads)
    Z2 = Z2 @ parameters['W_O2']

    X = layer_norm(X + Z2, parameters['gamma2'], parameters['beta2'])

    ff_out = feed_forward(X, parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2'])

    return layer_norm(X + ff_out, parameters['gamma3'], parameters['beta3'])

# Initializes the parameters for a Transformer model with n encoder and decoder blocks
def init_transformer(n, key):

    transformer_parameters = {}

    key, embed_key = random.split(key)
    transformer_parameters['E']= init_embedding(embed_key)

    key, enc_key = random.split(key)
    enc_keys = random.split(enc_key, n)
    key, dec_key = random.split(key)
    dec_keys = random.split(dec_key, n)

    encoder_parameters = [init_encoder(e_k) for e_k in enc_keys]
    decoder_parameters = [init_decoder(d_k) for d_k in dec_keys]

    key, out_key = random.split(key)
    keys = random.split(out_key, 2)

    transformer_parameters['W_out'] = random.normal(keys[0], (d_model, n_vocab)) * (1.0 / jnp.sqrt(d_model))
    transformer_parameters['b_out'] = random.normal(keys[1], (n_vocab,)) * (1.0 / jnp.sqrt(d_model))

    parameters = {
        'enc': encoder_parameters,
        'dec': decoder_parameters,
        'trans': transformer_parameters
    }

    return parameters

# Implements a Transformer model
@jax.jit
def transformer(src, tgt, parameters):

    encoder_parameters = parameters['enc']
    decoder_parameters = parameters['dec']
    transformer_parameters = parameters['trans']

    src = transformer_parameters['E'][src]
    tgt = transformer_parameters['E'][tgt]
    src = positional_encoding(src)

    for parameters in encoder_parameters:
        src = encoder_block(src, parameters)
    enc_out = src

    tgt = positional_encoding(tgt)

    for parameters in decoder_parameters:
        tgt = decoder_block(tgt, enc_out, parameters)

    return tgt @ transformer_parameters['W_out'] + transformer_parameters['b_out']

# Computes the cross entropy loss for a sample
def cross_entropy_loss(parameters, x, y):
    seq_len = y.shape[1]
    tgt_in = jnp.concatenate((jnp.zeros((y.shape[0], 1), dtype=y.dtype), y[:, :-1]), axis=1)
    logits = transformer(x, tgt_in, parameters)

    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, y[..., None], axis=-1).squeeze(-1)
    return jnp.mean(nll)

grad_loss = jax.grad(cross_entropy_loss)

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

# Performs a training step, returning the loss and updated parameters
@jax.jit
def train_step(parameters, x, y):
  loss = cross_entropy_loss(parameters, x, y)
  grads = grad_loss(parameters, x, y)
  parameters = jax.tree_util.tree_map(lambda p, g: p - lr * g, parameters, grads)
  return loss, parameters

key, X_key = random.split(key)
# Range is [1, n_vocab) since 0 is the start token
X = random.randint(X_key, (n_samples, seq_len), 1, n_vocab)
y = X[:, ::-1]

batches = batcher(X, y, batch_size, key)

key, init_key = random.split(key)
params = init_transformer(n_layers, init_key)

for epoch in range(epochs):

    for X_batch, y_batch in batches:

        loss, params = train_step(params, X_batch, y_batch)

    if epoch == 0 or (epoch+1) % 20 == 0:
      print(f"Epoch {epoch+1}, Loss: {loss:.3f}")