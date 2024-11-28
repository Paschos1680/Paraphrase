import torch
import json
import os
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset

# Paths to your models
pretrained_model_name = "facebook/bart-large"
fine_tuned_model_path = "C:/Users/Michalis/Desktop/ceid/HugginFace/FinalBART/checkpoint-7450"

# Directory to save attention matrices
save_directory = "C:/Users/Michalis/Desktop/ceid/HugginFace/Github/ExplainingBart/mm"

# Ensure the directory exists
os.makedirs(save_directory, exist_ok=True)

# Load tokenizer and models with the eager attention implementation
tokenizer = BartTokenizer.from_pretrained(pretrained_model_name)
pretrained_model = BartForConditionalGeneration.from_pretrained(
    pretrained_model_name, output_attentions=True, output_hidden_states=True, attn_implementation="eager"
)
fine_tuned_model = BartForConditionalGeneration.from_pretrained(
    fine_tuned_model_path, output_attentions=True, output_hidden_states=True, attn_implementation="eager"
)

# Load MRPC dataset and use the first sentence of the test split
mrpc_test = load_dataset("glue", "mrpc", split="test")
first_sentence = mrpc_test[0]["sentence1"]

# Function to extract intermediate matrices
def extract_attention_components(model, tokenizer, sentence, output_prefix):
    inputs = tokenizer(sentence, return_tensors="pt")

    # Pass inputs through the model
    outputs = model(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
    
    # Intermediate components
    decoder_hidden_states = outputs.decoder_hidden_states[-1]  # Decoder output
    attentions = outputs.decoder_attentions[-1]  # Last decoder layer attention

    # Extract weights for Q, K, V, and final projection
    wq = model.model.decoder.layers[-1].self_attn.q_proj.weight  # Query projection weight
    wk = model.model.decoder.layers[-1].self_attn.k_proj.weight  # Key projection weight
    wv = model.model.decoder.layers[-1].self_attn.v_proj.weight  # Value projection weight
    wo = model.model.decoder.layers[-1].self_attn.out_proj.weight  # Output projection weight

    # Compute Q, K, V as PyTorch tensors
    Q = torch.matmul(decoder_hidden_states, wq.T)  # Query
    K = torch.matmul(decoder_hidden_states, wk.T)  # Key
    V = torch.matmul(decoder_hidden_states, wv.T)  # Value

    # Attention weights (softmax of Q @ K.T)
    attn_weights = torch.nn.functional.softmax(torch.matmul(Q, K.transpose(-1, -2)), dim=-1)

    # Head output (attn @ V)
    head_out = torch.matmul(attn_weights, V)

    # Final output projection (head_out @ wO)
    final_output = torch.matmul(head_out, wo.T)

    # Save matrices as JSON
    components = {
        "Q": Q.detach().numpy(),
        "K": K.detach().numpy(),
        "V": V.detach().numpy(),
        "attention_weights": attn_weights.detach().numpy(),
        "head_out": head_out.detach().numpy(),
        "final_output": final_output.detach().numpy(),
    }

    for component_name, matrix in components.items():
        file_path = f"{save_directory}/{output_prefix}_{component_name}.json"
        with open(file_path, "w") as f:
            json.dump(matrix.tolist(), f)
        print(f"Saved {component_name} matrix to {file_path}.")

# Extract and save for both models
extract_attention_components(pretrained_model, tokenizer, first_sentence, "pretrained")
extract_attention_components(fine_tuned_model, tokenizer, first_sentence, "fine_tuned")
