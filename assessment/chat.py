import random
import json
import torch
from .model1 import NeuralNet
from .nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("assessment/intents.json", 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "assessment/data.pth"
data = torch.load(FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.60:
        for intent in intents['intents']:
            if tag == intent.get("tag"):  # Use .get() for safer dictionary access
                # **THE FIX IS HERE:**
                # Check if the 'responses' key exists and is not empty before choosing one.
                if intent.get("responses"):
                    return random.choice(intent.get("responses"))
                else:
                    # This handles cases where an intent is missing the "responses" key.
                    return f"I've found a topic ('{tag}') related to your query, but I don't have a specific response for it yet."

    return "I'm sorry, I'm still learning and don't quite understand. You can ask me about topics like stress, anxiety, depression, or how to support a friend."