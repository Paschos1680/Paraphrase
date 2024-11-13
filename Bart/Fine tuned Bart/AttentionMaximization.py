import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import matplotlib.pyplot as plt

# Set up model and tokenizer paths
model_path = r"C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450"
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)
model.eval()

# Specify the input sentence
text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."
inputs = tokenizer(text, return_tensors="pt")

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Convert input_ids to embeddings and make it a leaf variable with requires_grad enabled
inputs_embeds = model.get_input_embeddings()(inputs['input_ids']).clone().detach().requires_grad_(True)
decoder_input_ids = model.prepare_decoder_input_ids_from_labels(inputs["input_ids"])

# Forward pass with embeddings and decoder input ids to get outputs
outputs = model(inputs_embeds=inputs_embeds, decoder_input_ids=decoder_input_ids, output_hidden_states=True, return_dict=True)
last_hidden_state = outputs.decoder_hidden_states[-1]  # Get the last layer's hidden state

# Compute gradients for each neuron in the last layer
target_token_index = -1  # Typically, choose the last token in sequence
loss = last_hidden_state[0, target_token_index, :].sum()  # Sum all activations
loss.backward()

# Extract the gradients
importance_scores = inputs_embeds.grad.abs().mean(dim=1).squeeze().cpu().detach().numpy()

# Sort neurons by importance and select top neurons
importance_scores = importance_scores.flatten()
top_neurons = importance_scores.argsort()[-10:][::-1]  # Top 10 neurons

# Display results
print(f"Top 10 Important Neurons (by index): {top_neurons}")
print("Importance Scores of Top Neurons:", importance_scores[top_neurons])

# Plot the importance scores for visualization
plt.figure(figsize=(10, 6))
plt.bar(range(10), importance_scores[top_neurons])
plt.xticks(range(10), top_neurons)
plt.xlabel("Neuron Index")
plt.ylabel("Importance Score")
plt.title("Top 10 Neuron Importance Scores in Final Layer")
plt.show()

