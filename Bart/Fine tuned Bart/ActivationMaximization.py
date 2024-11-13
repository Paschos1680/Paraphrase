import torch
import torch.optim as optim
from transformers import BartTokenizer, BartForConditionalGeneration
import matplotlib.pyplot as plt

# Load model and tokenizer
model_path = r"C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450"
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Select a target neuron (or layer) to maximize activation
target_layer = model.model.encoder.layers[-1]  # Last encoder layer
target_neuron_index = 100  # Example neuron index to maximize; adjust as needed

# Define the target text input
initial_text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."

# Tokenize the initial text and get input embeddings
inputs = tokenizer(initial_text, return_tensors="pt")
input_ids = inputs['input_ids']
inputs_embedded = model.model.encoder.embed_tokens(input_ids).clone().detach()
inputs_embedded.requires_grad = True

# Define optimizer
optimizer = optim.Adam([inputs_embedded], lr=0.01)

# Number of optimization steps
num_steps = 200

# Run the optimization to maximize activation
for step in range(num_steps):
    optimizer.zero_grad()
    
    # Forward pass through the encoder
    outputs = model.model.encoder(inputs_embeds=inputs_embedded, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]  # Get activations from the last encoder layer
    
    # Target the activation of the specific neuron across all tokens
    target_activation = hidden_states[:, :, target_neuron_index].mean()
    
    # Maximize the target activation
    loss = -target_activation
    loss.backward()
    optimizer.step()
    
    # Optional: print progress every 50 steps
    if step % 50 == 0:
        print(f"Step {step}/{num_steps}, Activation Maximization Loss: {loss.item()}")

# Detach the optimized input embeddings to get the final activations
optimized_embeddings = inputs_embedded.detach()

# Final forward pass to get optimized activations for the target neuron
outputs = model.model.encoder(inputs_embeds=optimized_embeddings, output_hidden_states=True)
final_hidden_states = outputs.hidden_states[-1]
final_activations = final_hidden_states[:, :, target_neuron_index].squeeze().detach().cpu().numpy()

# Plot the activations of the optimized input for the target neuron
plt.figure(figsize=(10, 4))
plt.plot(final_activations, label="Optimized Activations")
plt.xlabel("Token Index")
plt.ylabel("Activation")
plt.title(f"Activation Maximization for Neuron {target_neuron_index} in Final Encoder Layer")
plt.legend()
plt.show()
