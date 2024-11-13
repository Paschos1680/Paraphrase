import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration

# Load models and tokenizer
pretrained_model_path = "facebook/bart-large"  # Path for pretrained BART
fine_tuned_model_path = r"C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450"

tokenizer = BartTokenizer.from_pretrained(pretrained_model_path)
pretrained_model = BartForConditionalGeneration.from_pretrained(pretrained_model_path)
fine_tuned_model = BartForConditionalGeneration.from_pretrained(fine_tuned_model_path)

# Ensure models are on the same device
device = torch.device("cpu")
pretrained_model.to(device)
fine_tuned_model.to(device)

# Define input text and preprocess
text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."
inputs = tokenizer(text, return_tensors="pt").to(device)

# Function to calculate layer-wise weight differences
def calculate_weight_diffs(pretrained_model, fine_tuned_model):
    weight_diffs = {}
    norms_pretrained = []
    norms_fine_tuned = []
    
    for (name1, param1), (name2, param2) in zip(pretrained_model.named_parameters(), fine_tuned_model.named_parameters()):
        if param1.requires_grad and "weight" in name1:
            diff = torch.abs(param1 - param2).mean().item()  # Average absolute difference
            weight_diffs[name1] = diff
            
            norms_pretrained.append(param1.norm().item())
            norms_fine_tuned.append(param2.norm().item())
    
    return weight_diffs, norms_pretrained, norms_fine_tuned

# Calculate weight differences and norms
weight_diffs, norms_pretrained, norms_fine_tuned = calculate_weight_diffs(pretrained_model, fine_tuned_model)

# Plot layer-wise weight differences
plt.figure(figsize=(12, 6))
plt.bar(weight_diffs.keys(), weight_diffs.values())
plt.xticks(rotation=90)
plt.title("Average Weight Difference by Layer")
plt.xlabel("Layer")
plt.ylabel("Average Absolute Weight Difference")
plt.tight_layout()
plt.show()

# Plot norms for pretrained vs fine-tuned
plt.figure(figsize=(10, 6))
plt.plot(norms_pretrained, label="Pretrained Model L2 Norm")
plt.plot(norms_fine_tuned, label="Fine-Tuned Model L2 Norm")
plt.title("L2 Norms of Weights by Layer")
plt.xlabel("Layer")
plt.ylabel("L2 Norm")
plt.legend()
plt.show()

# Neuron Activation Patterns
def calculate_activations(model, inputs, layer_idx):
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        activations = outputs.hidden_states[layer_idx].squeeze().cpu().numpy()
    return activations.mean(axis=0)

# Select a layer to compare neuron activations
layer_idx = -1  # last layer
activations_pretrained = calculate_activations(pretrained_model, inputs, layer_idx)
activations_fine_tuned = calculate_activations(fine_tuned_model, inputs, layer_idx)

# Plot neuron activations
plt.figure(figsize=(12, 6))
plt.plot(activations_pretrained, label="Pretrained Model")
plt.plot(activations_fine_tuned, label="Fine-Tuned Model")
plt.title(f"Neuron Activations in Layer {layer_idx} (Averaged)")
plt.xlabel("Neuron")
plt.ylabel("Activation")
plt.legend()
plt.show()
