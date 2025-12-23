import jax.numpy as jnp
from jax import random

# Initializes the embedding matrix
def init_embedding(key, n_vocab, d_model):
    return random.normal(key, (n_vocab, d_model)) * (1.0 / jnp.sqrt(d_model))

# Initializes the parameters for an encoder block
def init_encoder(key, d_model, n_hidden):

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

# Initializes the parameters for a decoder block
def init_decoder(key, d_model, n_hidden):

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


# Initializes the parameters for a Transformer model with n encoder and decoder blocks
def init_transformer(key, n_layers, n_vocab, d_model, n_hidden):

    transformer_parameters = {}

    key, embed_key = random.split(key)
    transformer_parameters['E']= init_embedding(embed_key, n_vocab, d_model)

    key, enc_key = random.split(key)
    enc_keys = random.split(enc_key, n_layers)
    key, dec_key = random.split(key)
    dec_keys = random.split(dec_key, n_layers)

    encoder_parameters = [init_encoder(e_k, d_model, n_hidden) for e_k in enc_keys]
    decoder_parameters = [init_decoder(d_k, d_model, n_hidden) for d_k in dec_keys]

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