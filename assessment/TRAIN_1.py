import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# These relative imports will work correctly when run as a module.
from .nltk_utils import bag_of_words, tokenize, stem
from .model1 import NeuralNet

# Load the comprehensive knowledge base
with open('assessment/intents.json', encoding='utf-8', errors='ignore') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Preprocess data
ignore_words = ['?', '.', '!']

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stemming and removing duplicates
all_words = sorted(set(stem(w) for w in all_words if w not in ignore_words))
tags = sorted(set(tags))

# Prepare training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)

# Hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)


# Create dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).type(torch.LongTensor)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Create data loader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#
# --- CUDA and Device Setup ---
#
# This line automatically selects the GPU if available, otherwise defaults to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on device: {device}')  # Added to show which device is being used

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        # Move data to the selected device (GPU or CPU)
        words = words.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

# Save trained model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "assessment/data.pth"
torch.save(data, FILE)

print(f'Training complete. Model saved to {FILE}')