import jax.numpy as jnp
from jax import random 
from .layers import multi_head_attention, layer_norm, feed_forward, positional_encoding, dropout

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
def decoder_block(X, num_heads, parameters, mask=None, cross=True, enc_output=None, dropout_rate=0.0, key=None):

    if dropout_rate > 0.0:
        if cross:
            keys = random.split(key, 5)
        else:
            keys = random.split(key, 3)

    Q = X @ parameters['W_Q']
    K = X @ parameters['W_K']
    V = X @ parameters['W_V']

    causal_mask = jnp.triu(jnp.full((1, 1, X.shape[1], X.shape[1]), -1e9), k=1)
    if mask is not None:
        causal_mask += mask

    Z = multi_head_attention(
        Q, K, V, num_heads, mask=causal_mask, 
        dropout_rate=dropout_rate, 
        key=keys[0] if dropout_rate > 0.0 else None)
    Z = Z @ parameters['W_O']

    if dropout_rate > 0.0:
        Z = dropout(Z, dropout_rate, keys[1])

    X = layer_norm(X + Z, parameters['gamma1'], parameters['beta1'])

    if cross:
        Q2 = X @ parameters['W_Q2']
        K2 = enc_output @ parameters['W_K2']
        V2 = enc_output @ parameters['W_V2']

        Z2 = multi_head_attention(
            Q2, K2, V2, num_heads, dropout_rate=dropout_rate,
            key=keys[3] if dropout_rate > 0.0 else None)
        
        Z2 = Z2 @ parameters['W_O2']
        if dropout_rate > 0.0:
            Z2 = dropout(Z2, dropout_rate, keys[4])

        X = layer_norm(X + Z2, parameters['gamma2'], parameters['beta2'])

    ff_out = feed_forward(X, parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2'])
    if dropout_rate > 0.0:
        ff_out = dropout(ff_out, dropout_rate, keys[2])

    return layer_norm(X + ff_out, parameters['gamma3'], parameters['beta3'])

# Implements an encoder-decoder Transformer
def encoder_decoder_transformer(src, tgt, num_heads, parameters, mask=None):

    encoder_parameters = parameters['enc']
    decoder_parameters = parameters['dec']
    transformer_parameters = parameters['trans']

    src = transformer_parameters['E'][src]
    tgt = transformer_parameters['E'][tgt]
    src = positional_encoding(src)

    for block_parameters in encoder_parameters:
        src = encoder_block(src, num_heads, block_parameters)
    enc_out = src

    tgt = positional_encoding(tgt)

    for block_parameters in decoder_parameters:
        tgt = decoder_block(tgt, num_heads, block_parameters, mask=mask, cross=True, enc_output=enc_out)

    return tgt @ transformer_parameters['W_out'] + transformer_parameters['b_out']

# Implements a decoder-only Transformer
def decoder_only_transformer(tgt, num_heads, parameters, mask=None, dropout_rate=0.0, key=None):

    decoder_parameters = parameters['dec']
    transformer_parameters = parameters['trans']

    tgt = transformer_parameters['E'][tgt]
    tgt = positional_encoding(tgt)

    if dropout_rate > 0.0:
        key, emb_key = random.split(key)
        tgt = dropout(tgt, dropout_rate, emb_key)

    for block_parameters in decoder_parameters:
        if dropout_rate > 0.0:
            key, block_key = random.split(key)
        tgt = decoder_block(
            tgt, num_heads, block_parameters, mask=mask, cross=False,
            dropout_rate=dropout_rate, key=block_key if dropout_rate > 0.0 else None)

    return tgt @ transformer_parameters['W_out'] + transformer_parameters['b_out']