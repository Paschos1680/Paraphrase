# Import necessary libraries
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from captum.attr import LayerConductance
import matplotlib.pyplot as plt

# Load BART model and tokenizer
model_path = r"C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450" 
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path).eval().to("cuda" if torch.cuda.is_available() else "cpu")

# Prepare input text
text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generate embeddings from input_ids and set up 4D attention mask
hidden_states = model.model.encoder.embed_tokens(inputs['input_ids'].long()) * model.model.encoder.embed_scale
batch_size, seq_length = inputs['input_ids'].shape
attention_mask = inputs['attention_mask'].unsqueeze(1).unsqueeze(2).repeat(1, 1, seq_length, 1)  # Shape: (batch, 1, seq, seq)

# Define target output token index for Layer Conductance
target_token_index = -1  # Last token in the sequence

# Custom forward function for Layer Conductance
def custom_forward(hidden_states, attention_mask, layer):
    layer_head_mask = torch.ones(16).to(model.device)
    output = layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask
    )
    return output[0][..., target_token_index]

# Conductance function
def compute_layer_conductance(hidden_states, attention_mask, layer, target_token_index):
    try:
        conductance = LayerConductance(custom_forward, layer)
        attributions = conductance.attribute(
            hidden_states,
            additional_forward_args=(attention_mask, layer),
            target=target_token_index
        )
        return attributions
    except Exception as e:
        print("Error in Layer Conductance:", e)
        return None

# Mean Conductance Plot (Approach 1)
conductance_attributions = compute_layer_conductance(hidden_states, attention_mask, model.model.encoder.layers[-1], target_token_index)
if conductance_attributions is not None:
    conductance_scores_mean = conductance_attributions.mean(dim=-1).squeeze().detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    plt.figure(figsize=(10, 5))
    plt.bar(tokens, conductance_scores_mean)
    plt.xlabel("Tokens")
    plt.ylabel("Mean Layer Conductance Scores")
    plt.title("Mean Layer Conductance for Each Token (Encoder)")
    plt.xticks(rotation=45)
    plt.savefig("mean_conductance.png")
    plt.show()

# Layer-by-Layer Visualization (Approach 2) - 12-layer grid for encoder
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("Layer Conductance for Each Encoder Layer")
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

for i, layer in enumerate(model.model.encoder.layers):
    conductance_attributions = compute_layer_conductance(hidden_states, attention_mask, layer, target_token_index)
    conductance_scores = conductance_attributions.mean(dim=-1).squeeze().detach().cpu().numpy()
    ax = axes[i // 4, i % 4]
    ax.bar(tokens, conductance_scores)
    ax.set_title(f"Encoder Layer {i + 1}")
    ax.set_xticklabels(tokens, rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("encoder_layers_conductance.png")
plt.show()

# Repeat for Decoder if needed (Approach 2) - 12-layer grid for decoder
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("Layer Conductance for Each Decoder Layer")

for i, layer in enumerate(model.model.decoder.layers):
    conductance_attributions = compute_layer_conductance(hidden_states, attention_mask, layer, target_token_index)
    conductance_scores = conductance_attributions.mean(dim=-1).squeeze().detach().cpu().numpy()
    ax = axes[i // 4, i % 4]
    ax.bar(tokens, conductance_scores)
    ax.set_title(f"Decoder Layer {i + 1}")
    ax.set_xticklabels(tokens, rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("decoder_layers_conductance.png")
plt.show()

# Top-K Important Dimensions Plot (Approach 3)
# Extract and sort top K dimensions for visualization
K = 10  # Adjust K as needed
top_k_scores = conductance_attributions.topk(K, dim=-1)[0].mean(dim=1).cpu().numpy()
plt.figure(figsize=(10, 5))
plt.bar(tokens, top_k_scores)
plt.xlabel("Tokens")
plt.ylabel(f"Top-{K} Mean Layer Conductance Scores")
plt.title(f"Top-{K} Layer Conductance for Each Token")
plt.xticks(rotation=45)
plt.savefig("top_k_conductance.png")
plt.show()
