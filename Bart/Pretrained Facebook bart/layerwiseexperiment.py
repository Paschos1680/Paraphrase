import torch
import matplotlib.pyplot as plt
from captum.attr import LayerConductance
from transformers import BartTokenizer, BartForConditionalGeneration

# Load model and tokenizer
model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.eval()

# Text input
text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."

# Tokenize input with integer-based IDs
inputs = tokenizer(text, return_tensors="pt")

# Function for Layer Conductance for encoder
def compute_layer_conductance(model, inputs, target_layer):
    conductance = LayerConductance(model, target_layer)
    try:
        # Directly use input_ids and specify target as the first token (or adjust as needed)
        attributions = conductance.attribute(inputs['input_ids'], target=0)
        return attributions
    except Exception as e:
        print("Error in Layer Conductance:", e)
        return None

# Plot conductance for each encoder layer
fig, axes = plt.subplots(3, 4, figsize=(20, 10))
fig.suptitle("Layer Conductance for Each Encoder Layer")
for i, ax in enumerate(axes.flatten()):
    layer = model.model.encoder.layers[i]
    conductance_attributions = compute_layer_conductance(model, inputs, layer)
    
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
