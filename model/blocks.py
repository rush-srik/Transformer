import jax.numpy as jnp
from .layers import multi_head_attention, layer_norm, feed_forward, positional_encoding

# Implements a single encoder block
def encoder_block(X, num_heads, parameters):

    Q = X @ parameters['W_Q']
    K = X @ parameters['W_K']
    V = X @ parameters['W_V']

    Z = multi_head_attention(Q, K, V, num_heads)
    Z = Z @ parameters['W_O']

    X = layer_norm(X + Z, parameters['gamma1'], parameters['beta1'])

    ff_out = feed_forward(X, parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2'])

    return layer_norm(X + ff_out, parameters['gamma2'], parameters['beta2'])

# Implements a single decoder block
def decoder_block(X, enc_output, num_heads, parameters, mask=None):

    Q = X @ parameters['W_Q']
    K = X @ parameters['W_K']
    V = X @ parameters['W_V']

    causal_mask = jnp.triu(jnp.full((1, 1, X.shape[1], X.shape[1]), -1e9), k=1)
    if mask is not None:
        causal_mask += mask

    Z = multi_head_attention(Q, K, V, num_heads, mask=causal_mask)
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

# Implements an encoder-decoder Transformer
def encoder_decoder_transformer(src, tgt, num_heads, parameters, mask=None):

    encoder_parameters = parameters['enc']
    decoder_parameters = parameters['dec']
    transformer_parameters = parameters['trans']

    src = transformer_parameters['E'][src]
    tgt = transformer_parameters['E'][tgt]
    src = positional_encoding(src)

    for parameters in encoder_parameters:
        src = encoder_block(src, num_heads, parameters)
    enc_out = src

    tgt = positional_encoding(tgt)

    for parameters in decoder_parameters:
        tgt = decoder_block(tgt, enc_out, num_heads, parameters, mask=mask)

    return tgt @ transformer_parameters['W_out'] + transformer_parameters['b_out']