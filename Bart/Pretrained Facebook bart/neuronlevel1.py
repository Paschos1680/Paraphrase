import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import matplotlib.pyplot as plt

# Load the pretrained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the input text and tokenize it
text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."
inputs = tokenizer(text, return_tensors="pt").to(device)

# Dictionary to store neuron activations
neuron_activations = {}
importance_scores = {}

# Hook function to capture activations
def capture_activations(module, input, output):
    # Handle cases where output is a tuple
    if isinstance(output, tuple):
        output = output[0]
    layer_activations = output.detach().cpu().numpy().flatten()
    layer_name = f"{module}"
    if layer_name not in neuron_activations:
        neuron_activations[layer_name] = []
    neuron_activations[layer_name].append(layer_activations)

# Attach hooks to each encoder layer
for i, layer in enumerate(model.model.encoder.layers):
    layer.register_forward_hook(capture_activations)

# Forward pass to capture activations
with torch.no_grad():
    outputs = model(**inputs)

# Process activations for importance scores
for layer_name, activations in neuron_activations.items():
    activations_tensor = torch.tensor(activations)
    importance_scores[layer_name] = activations_tensor.mean(dim=0).numpy()  # Compute mean activation per neuron as importance

# Visualize neuron activations for each encoder layer
for layer_name, scores in importance_scores.items():
    plt.figure(figsize=(10, 4))
    plt.plot(scores)
    plt.title(f"Neuron Importance for {layer_name}")
    plt.xlabel("Neuron Index")
    plt.ylabel("Mean Activation (Importance Score)")
    plt.show()

