# Import necessary libraries
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np
import matplotlib.pyplot as plt

# Load the model and tokenizer
model_name = "facebook/bart-large"
model = BartForConditionalGeneration.from_pretrained(model_name, output_attentions=True)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Define the input text and encode
text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."
inputs = tokenizer(text, return_tensors="pt")

# Get attention from encoder
outputs = model(**inputs)
encoder_attentions = outputs.encoder_attentions

# Function to perform Attention Rollout
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
