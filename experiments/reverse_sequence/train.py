from experiments.utils import train_step, batcher, train_test_split, predict, accuracy, cross_entropy_loss
from model.init import init_transformer
from jax import random
import matplotlib.pyplot as plt

# Hyperparameters
seq_len = 8
d_model = 64
n_heads = 8
n_hidden = 4 * d_model
n_layers = 2
n_vocab = 10

n_samples = 10_000
batch_size = 128
epochs = 50
lr = 1e-1

key = random.key(42)

key, X_key = random.split(key)
# Range is [1, n_vocab) since 0 is the start token
X = random.randint(X_key, (n_samples, seq_len), 1, n_vocab)
y = X[:, ::-1]

key, split_key = random.split(key)
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2, split_key)

key, init_key = random.split(key)
params = init_transformer(init_key, n_layers, n_vocab, d_model, n_hidden)

train_losses =[]
test_losses = []
test_accs = []

for epoch in range(epochs):

    key, batch_key = random.split(key)
    batches = batcher(X_train, y_train, batch_size, batch_key)

    train_loss = 0
    for X_batch, y_batch in batches:

        loss, params = train_step(params, n_heads, X_batch, y_batch, lr)
        train_loss += loss

    train_loss /= (len(X_train) // batch_size)

    test_loss = cross_entropy_loss(params, n_heads, X_test, y_test)
    test_pred = predict(X_test, params, n_heads)
    test_acc = accuracy(y_test, test_pred)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    if epoch == 0 or (epoch+1) % 10 == 0:
      print(f'Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}')

sample_preds = predict(X_test[:3], params, n_heads)

for i in range(3):
  print(f'Input: {X_test[i]}')
  print(f'Predicted: {sample_preds[i]}')
  print(f'Actual: {y_test[i]}')

# Plot loss
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Testing loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.savefig('loss.png')

# Plot test accuracy
plt.plot(test_accs)
plt.title('Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.savefig('acc.png')

