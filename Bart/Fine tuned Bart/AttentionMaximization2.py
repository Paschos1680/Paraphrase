import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import matplotlib.pyplot as plt

# Choose model path: either custom fine-tuned BART or Facebook BART
# For custom BART:
model_path = r"C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450"
# For Facebook BART:
# model_path = "facebook/bart-large"

model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)
model.eval()

# Specify the input sentence
text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."
inputs = tokenizer(text, return_tensors="pt")

# Function to extract and plot neuron importance scores for a specified layer
def plot_neuron_importance(model, inputs, layer_index, layer_name=""):
    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])

    # Forward pass through the model
    encoder_outputs = model.model.encoder(inputs_embeds=inputs_embeds, output_hidden_states=True, return_dict=True)
    hidden_states = encoder_outputs.hidden_states[layer_index]

    # Calculate neuron importance by averaging over tokens
    importance_scores = hidden_states.mean(dim=1).squeeze().detach().cpu().numpy()

    # Plot importance scores
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importance_scores)), importance_scores)
    plt.xlabel("Neuron Index")
    plt.ylabel("Importance Score")
    plt.title(f"Neuron Importance - Layer {layer_index} {layer_name}")
    plt.show()

# Plot for the first, middle, and last layers
plot_neuron_importance(model, inputs, 0, "First Layer")
plot_neuron_importance(model, inputs, len(model.model.encoder.layers) // 2, "Middle Layer")
plot_neuron_importance(model, inputs, len(model.model.encoder.layers) - 1, "Last Layer")

