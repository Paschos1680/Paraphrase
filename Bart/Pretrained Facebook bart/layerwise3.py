import torch
import matplotlib.pyplot as plt
from captum.attr import LayerConductance
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np

# Load model and tokenizer
model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.eval()

# Text input
text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."

# Tokenize input
inputs = tokenizer(text, return_tensors="pt")
inputs_embedded = model.get_encoder()(inputs['input_ids'], return_dict=True)

# Define target token (optional - use for classification)
target_token_index = 0  # example, adapt as needed

# Function for Layer Conductance
def compute_layer_conductance(model, inputs_embedded, target=None):
    conductance = LayerConductance(model, model.model.encoder.layers[0])
    try:
        attributions = conductance.attribute(inputs_embedded['last_hidden_state'], target=target)
        return attributions
    except Exception as e:
        print("Error in Layer Conductance:", e)
        return None

# Compute and plot conductance for encoder layers
fig, axes = plt.subplots(3, 4, figsize=(20, 10))
fig.suptitle("Layer Conductance for Each Encoder Layer")
for i, ax in enumerate(axes.flatten()):
    layer = model.model.encoder.layers[i]
    conductance = LayerConductance(model, layer)
    conductance_attributions = compute_layer_conductance(model, inputs_embedded, target_token_index)

    if conductance_attributions is not None:
        conductance_scores = conductance_attributions.mean(dim=-1).squeeze().detach().cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
        ax.bar(range(len(tokens)), conductance_scores)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_title(f"Encoder Layer {i+1}")
    else:
        ax.set_title(f"Encoder Layer {i+1} (No Conductance)")

plt.tight_layout()
plt.show()

# Additional logic for decoder, if necessary
