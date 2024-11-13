# Import necessary libraries
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import matplotlib.pyplot as plt

# Load the BART model and tokenizer
model_path = "facebook/bart-large"  # Use Facebook BART
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path).eval()

# Move model to device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the input text
text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."
inputs = tokenizer(text, return_tensors="pt").to(device)

# Define dictionaries to store activations
encoder_activations = {}
decoder_activations = {}

# Function to create forward hook with handling for tuple outputs
def hook_fn(module, input, output, layer_id, activations_dict):
    # Extract hidden states from tuple output
    hidden_states = output[0] if isinstance(output, tuple) else output
    activations_dict[layer_id] = hidden_states.detach().cpu()

# Register hooks on the encoder and decoder layers
for i, layer in enumerate(model.model.encoder.layers):
    layer.register_forward_hook(lambda m, inp, out: hook_fn(m, inp, out, i, encoder_activations))

for i, layer in enumerate(model.model.decoder.layers):
    layer.register_forward_hook(lambda m, inp, out: hook_fn(m, inp, out, i, decoder_activations))

# Pass the inputs through the model to activate hooks
with torch.no_grad():
    outputs = model(**inputs)

# Visualize encoder layer activations
print("\n--- Encoder Layer Activations ---")
for layer_id, activation in encoder_activations.items():
    avg_activation = activation.mean(dim=1).squeeze().numpy()  # Average across batch dimension
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    plt.figure(figsize=(10, 5))
    plt.plot(avg_activation, label=f"Layer {layer_id} Activation")
    plt.xlabel("Tokens")
    plt.ylabel("Activation")
    plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=45)
    plt.title(f"Activation Pattern for Encoder Layer {layer_id}")
    plt.legend()
    plt.show()

# Visualize decoder layer activations
print("\n--- Decoder Layer Activations ---")
for layer_id, activation in decoder_activations.items():
    avg_activation = activation.mean(dim=1).squeeze().numpy()  # Average across batch dimension
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    plt.figure(figsize=(10, 5))
    plt.plot(avg_activation, label=f"Layer {layer_id} Activation")
    plt.xlabel("Tokens")
    plt.ylabel("Activation")
    plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=45)
    plt.title(f"Activation Pattern for Decoder Layer {layer_id}")
    plt.legend()
    plt.show()
