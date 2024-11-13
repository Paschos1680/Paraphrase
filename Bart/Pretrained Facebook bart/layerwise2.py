# Import necessary libraries
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from captum.attr import LayerConductance
import matplotlib.pyplot as plt

# Load the BART model and tokenizer (adjust model_path as needed)
model_path = "facebook/bart-large"  # or r"C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450" for fine-tuned
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path).eval().to("cuda" if torch.cuda.is_available() else "cpu")

# Prepare input text
text = "PCCW's chief operating officer, Mike Butcher, and Alex Arena, the chief financial officer, will report directly to Mr So."
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generate embeddings from input_ids and set up 4D attention mask
hidden_states = model.model.encoder.embed_tokens(inputs['input_ids']) * model.model.encoder.embed_scale
batch_size, seq_length = inputs['input_ids'].shape
attention_mask = inputs['attention_mask'].unsqueeze(1).unsqueeze(2).repeat(1, 1, seq_length, 1)  # Shape: (batch, 1, seq, seq)

# Custom forward function for Layer Conductance
def custom_forward_encoder(hidden_states, attention_mask, encoder_layer):
    """A wrapper to provide required arguments for Layer Conductance."""
    layer_head_mask = torch.ones(16).to(model.device)  # Adjusted to correct head size
    output = encoder_layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask
    )
    return output[0]

# Layer Conductance function
def compute_layer_conductance(hidden_states, attention_mask, encoder_layer):
    try:
        conductance = LayerConductance(custom_forward_encoder, encoder_layer)
        attributions = conductance.attribute(hidden_states, additional_forward_args=(attention_mask, encoder_layer))
        print("Layer Conductance computed successfully.")
        return attributions
    except Exception as e:
        print("Error in Layer Conductance:", e)
        return None

# Gradient-Based LRP Approximation
def compute_gradient_based_lrp(model, inputs):
    embeddings = model.model.encoder.embed_tokens(inputs['input_ids']).clone().detach().requires_grad_(True)
    inputs_embedded = {
        "inputs_embeds": embeddings,
        "attention_mask": inputs['attention_mask'],
        "decoder_input_ids": torch.tensor([[model.config.decoder_start_token_id]]).to(model.device)
    }
    outputs = model(**inputs_embedded)
    logits = outputs.logits[:, -1, :]  # Final output logit
    logits.backward(gradient=torch.ones_like(logits))
    relevance_scores = embeddings.grad.detach().cpu().numpy().squeeze()
    return relevance_scores

# Choose a specific encoder layer
encoder_layer = model.model.encoder.layers[-1]

# Calculate Layer Conductance
conductance_attributions = compute_layer_conductance(hidden_states, attention_mask, encoder_layer)

# Calculate Gradient-Based LRP and reduce dimensions
lrp_scores = compute_gradient_based_lrp(model, inputs)
lrp_scores_reduced = lrp_scores.mean(axis=1)  # Average across hidden dimensions to get per-token relevance

# Visualize Conductance Attributions
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
if conductance_attributions is not None:
    conductance_scores = conductance_attributions.squeeze().detach().cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.bar(tokens, conductance_scores)
    plt.xlabel("Tokens")
    plt.ylabel("Layer Conductance Scores")
    plt.title("Layer Conductance for Each Token (Encoder)")
    plt.xticks(rotation=45)
    plt.show()

# Visualize Gradient-Based LRP Scores
if lrp_scores is not None:
    plt.figure(figsize=(10, 5))
    plt.bar(tokens, lrp_scores_reduced)
    plt.xlabel("Tokens")
    plt.ylabel("LRP Approximation Scores (Gradients)")
    plt.title("Gradient-Based LRP Approximation for Each Token")
    plt.xticks(rotation=45)
    plt.show()

