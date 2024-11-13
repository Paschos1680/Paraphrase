import torch
import matplotlib.pyplot as plt
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np

# Load the Facebook BART model and tokenizer
model_path = "facebook/bart-large"
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)

# Define a sample text
text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."

# Tokenize input text
inputs = tokenizer(text, return_tensors="pt")
inputs_embedded = model.model.encoder.embed_tokens(inputs['input_ids'])

# Ensure gradients can be computed for inputs
inputs_embedded.requires_grad_(True)

# Select the last layer of the encoder for neuron importance analysis
last_layer = model.model.encoder.layers[-1]

# Function to compute neuron importance
def compute_neuron_importance(layer, inputs_embedded):
    # Forward pass through the encoder to get outputs
    outputs = model.model.encoder(inputs_embeds=inputs_embedded, output_hidden_states=True)
    hidden_states = outputs.last_hidden_state
    
    # Compute the importance scores by taking the gradient of the mean of the hidden states
    importance_scores = torch.autograd.grad(hidden_states.mean(), inputs_embedded)[0]
    # Aggregate importance scores by taking the mean across all tokens
    importance_scores_mean = importance_scores.mean(dim=1).squeeze().detach().cpu().numpy()
    return importance_scores_mean

# Compute neuron importance
importance_scores = compute_neuron_importance(last_layer, inputs_embedded)

# Plot the neuron importance scores
plt.figure(figsize=(12, 4))
plt.bar(range(len(importance_scores)), importance_scores)
plt.xlabel("Neuron Index")
plt.ylabel("Importance Score")
plt.title("Neuron Importance Attribution for Final Encoder Layer in Facebook BART")
plt.show()
