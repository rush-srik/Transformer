import requests
import jax
import jax.numpy as jnp
from jax import random
import pickle
from experiments.utils import init_adam_state, adam_step, cross_entropy_loss, get_lr, predict_dec_only
from model.init import init_dec_transformer
import matplotlib.pyplot as plt
from tqdm import tqdm

data = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt').text

chars = sorted(list(set(data)))

# Hyperparameters
block_size = 512
d_model = 256
n_heads = 8
n_hidden = 4 * d_model
n_layers = 6
n_vocab = len(chars) + 1

batch_size = 64
steps = 5000
base_lr = 3e-4
min_lr = 0.1 * base_lr
warmup_steps = 0.1 * steps
dropout_rate = 0.2
weight_decay = 1e-2

key = random.key(42)

stoi = {ch:i+1 for i, ch in enumerate(chars)}
stoi['<start>'] = 0
itos = {i:ch for ch, i in stoi.items()}

data = jnp.array([stoi[c] for c in data])

train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]

# Makes a dataset of input-output pairs
def make_slices(data, block_size):
    start_idx = jnp.arange(0, len(data) - block_size, block_size)

    def get_slice(idx):
      return jax.lax.dynamic_slice(data, (idx,), (block_size+1,))

    return jax.vmap(get_slice)(start_idx)

train_slices = make_slices(train_data, block_size)
test_slices = make_slices(test_data, block_size)

print(''.join([itos[int(i)] for i in train_slices[0]]))

# Samples a batch from the dataset
def sample_batch(slices, batch_size, key):
  idx = random.randint(key, (batch_size,), 0, len(slices)-1)
  return slices[idx]

key, init_key = random.split(key)
params = init_dec_transformer(init_key, n_layers, n_vocab, d_model, n_hidden)

m, v, t = init_adam_state(params)

train_losses =[]
test_losses = []

for step in tqdm(range(steps)):

    key, batch_key = random.split(key)
    batch = sample_batch(train_slices, batch_size, batch_key)

    lr = get_lr(step, steps, base_lr, min_lr, warmup_steps)
    step_key, key = random.split(key)
    train_loss, params, m, v, t = adam_step(params, n_heads, batch, lr, m, v, t, weight_decay)

    train_losses.append(train_loss)

    if step == 0 or (step+1) % 1000 == 0:

      key, batch_key = random.split(key)
      test_batch = sample_batch(test_slices, batch_size, batch_key)

      test_loss = cross_entropy_loss(params, n_heads, test_batch, encdec=False)
      test_losses.append(test_loss)
      print(f' Step {step+1} | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f} | Perplexity: {jnp.exp(test_loss):.3f}')

plt.plot(train_losses, label='Training loss')
plt.plot(jnp.arange(0, steps+1000, 1000), test_losses, label='Testing loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Loss')

plt.savefig('loss.png')

start_str = 'ROMEO:\n'
start_ids = [stoi[c] for c in start_str]

gen_key, key = random.split(key)
output_ids = predict_dec_only(params, n_heads, block_size, 
                              max_length=10000, temperature=0.8, start_tokens=start_ids, key=key)

print(''.join([itos[i] for i in output_ids]))