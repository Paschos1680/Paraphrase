import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# Load the BART model and tokenizer with attn_implementation="eager" to avoid warnings
model_name = "facebook/bart-large"
model = BartForConditionalGeneration.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")
tokenizer = BartTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Sample input text
text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."
inputs = tokenizer(text, return_tensors="pt").to(device)

# Forward pass to capture attentions
outputs = model(**inputs, output_attentions=True)
encoder_attentions = outputs.encoder_attentions  # Encoder self-attention
decoder_attentions = outputs.decoder_attentions  # Decoder self-attention
cross_attentions = outputs.cross_attentions      # Cross-attention

# Get token list for visualization
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Custom function for attention map visualization
def display_custom_attention_map(attentions, tokens, layer_num=0, head_num=0):
    print("\n--- Custom Attention Maps ---")
    selected_attention = attentions[layer_num][0]  # First batch element, specific layer

    # Choose a specific head for visualization
    head_attention = selected_attention[head_num].detach().cpu().numpy()  # Specific head

    plt.figure(figsize=(10, 8))
    plt.imshow(head_attention, cmap="viridis")
    plt.colorbar()
    plt.title(f"Attention Map - Layer {layer_num + 1}, Head {head_num + 1}")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.show()

# Display custom attention map for encoder's first layer and first head
display_custom_attention_map(encoder_attentions, tokens, layer_num=0, head_num=0)

# 2. Attention Rollout (Cumulative Attention Across Layers)
def attention_rollout(attentions):
    print("\n--- Attention Rollout ---")
    cumulative_attention = attentions[0].mean(dim=1)  # Start with first layer's average attention
    for layer_attention in attentions[1:]:
        avg_attention = layer_attention.mean(dim=1)
        cumulative_attention = torch.matmul(cumulative_attention, avg_attention)
    cumulative_attention_np = cumulative_attention[0].detach().cpu().numpy()
    
    plt.imshow(cumulative_attention_np, cmap="viridis")
    plt.colorbar()
    plt.title("Attention Rollout - Cumulative Attention")
    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.show()

# Generate and plot attention rollout for encoder
attention_rollout(encoder_attentions)

# 3. Head-Specific Attention Analysis (Visualize Specific Head in a Layer)
def display_head_specific_attention(attentions, tokens, layer_num=0, head_num=0):
    print("\n--- Head-Specific Attention ---")
    layer_attention = attentions[layer_num][0]  # First element in batch, specified layer
    specific_head_attention = layer_attention[head_num]  # Specific head
    plt.imshow(specific_head_attention.detach().cpu().numpy(), cmap="viridis")
    plt.title(f"Layer {layer_num + 1}, Head {head_num + 1}")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.colorbar()
    plt.show()

# Example visualization of head-specific attention for encoder
display_head_specific_attention(encoder_attentions, tokens, layer_num=0, head_num=0)

# 4. Attention-Based Clustering (Group Tokens by Attention Patterns)
def attention_clustering(attentions, tokens, n_clusters=3):
    print("\n--- Attention-Based Clustering ---")
    last_layer_attention = attentions[-1][0].mean(dim=0).cpu().detach().numpy()  # Average heads in last layer
    pca = PCA(n_components=2)
    reduced_attention = pca.fit_transform(last_layer_attention)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(reduced_attention)
    plt.scatter(reduced_attention[:, 0], reduced_attention[:, 1], c=kmeans.labels_, cmap="viridis")
    for i, token in enumerate(tokens):
        plt.annotate(token, (reduced_attention[i, 0], reduced_attention[i, 1]))
    plt.title("Attention-Based Clustering")
    plt.show()

# Run attention clustering for encoder attentions
attention_clustering(encoder_attentions, tokens, n_clusters=3)








