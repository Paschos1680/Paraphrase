# Import necessary libraries
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from bertviz import head_view
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Load the fine-tuned model and tokenizer
model_path = r"C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450"
model = BartForConditionalGeneration.from_pretrained(model_path, output_attentions=True)
tokenizer = BartTokenizer.from_pretrained(model_path)

# Define the input text and encode
text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Get attention data
encoder_attentions = outputs.encoder_attentions
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Attention Maps - Visualize one head in the first layer
def display_attention_map(attentions, tokens, layer_num=0, head_num=0):
    attention = attentions[layer_num][0, head_num].detach().cpu().numpy()
    plt.matshow(attention, cmap="viridis")
    plt.colorbar()
    plt.title(f"Attention Map - Layer {layer_num+1}, Head {head_num+1}")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.show()

# Display the attention map for the first head in the first layer
display_attention_map(encoder_attentions, tokens, layer_num=0, head_num=0)

# Head-Specific Attention Analysis
def head_specific_analysis(attentions, tokens, layer_num=0):
    for head_num in range(attentions[layer_num].size(1)):
        attention = attentions[layer_num][0, head_num].detach().cpu().numpy()
        plt.matshow(attention, cmap="viridis")
        plt.title(f"Layer {layer_num+1} Head {head_num+1}")
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
        plt.show()

# Head-specific analysis for layer 1
head_specific_analysis(encoder_attentions, tokens, layer_num=0)

# Attention-Based Clustering
def attention_based_clustering(attentions, tokens, layer_num=0):
    attention = attentions[layer_num].mean(dim=1)[0].detach().cpu().numpy()
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(attention)
    plt.scatter(attention[:, 0], attention[:, 1], c=clusters, cmap="viridis")
    for i, token in enumerate(tokens):
        plt.text(attention[i, 0], attention[i, 1], token, fontsize=12)
    plt.title("Attention-Based Clustering")
    plt.show()

# Perform clustering on the first layer
attention_based_clustering(encoder_attentions, tokens, layer_num=0)

# Attention Rollout
def attention_rollout(attentions):
    rollout = torch.eye(attentions[0].size(-1)).unsqueeze(0).to(attentions[0].device)
    for attention in attentions:
        attention_heads_mean = attention.mean(dim=1)  # Average across heads
        rollout = torch.matmul(attention_heads_mean, rollout)
    return rollout[0].detach().cpu().numpy()

# Perform Attention Rollout
rollout_attention = attention_rollout(encoder_attentions)

# Plot the rollout attention
plt.imshow(rollout_attention, cmap="viridis")
plt.colorbar()
plt.title("Attention Rollout")
plt.xlabel("Tokens")
plt.ylabel("Tokens")
plt.show()
