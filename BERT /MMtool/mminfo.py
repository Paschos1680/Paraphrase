import torch
import numpy as np
import json
from transformers import BertTokenizer, BertModel

# Paths
fine_tuned_model_path = "C:\\Users\\Michalis\\Desktop\\ceid\\HugginFace\\final_model_bert"
save_directory = "C:\\Users\\Michalis\\Desktop\\ceid\\HugginFace\\Github\\Explaining BERT"

# Load Models
fine_tuned_model = BertModel.from_pretrained(fine_tuned_model_path, output_attentions=True, output_hidden_states=True)
bert_base_model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True, output_hidden_states=True)

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# MRPC Test Sentences
sentence1 = "He said the food was good."
sentence2 = "The man mentioned the meal was delicious."

# Prepare Input for Paraphrase Checking
inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)

# Function to Serialize Safely
def serialize_component(component):
    """Convert tensors, arrays, tuples, and lists to JSON-serializable formats."""
    if isinstance(component, torch.Tensor):
        return component.detach().cpu().numpy().tolist()
    elif isinstance(component, np.ndarray):
        return component.tolist()
    elif isinstance(component, tuple):  # Handle tuples by serializing each element
        return [serialize_component(c) for c in component]
    elif isinstance(component, list):
        return [serialize_component(c) for c in component]
    else:
        raise TypeError(f"Unsupported type: {type(component)}")

# Function to Extract Features During Operation
def extract_bert_features_during_operation(model, inputs, output_prefix):
    outputs = model(**inputs)
    
    # Extract outputs
    attention_weights = outputs.attentions  # Attention weights
    hidden_states = outputs.hidden_states  # Hidden states for all layers
    final_output = outputs.last_hidden_state  # Final layer output (last hidden state)

    # Extract Q, K, V, Head Outputs
    last_layer_hidden = hidden_states[-1]
    query_proj = model.encoder.layer[-1].attention.self.query.weight.T
    key_proj = model.encoder.layer[-1].attention.self.key.weight.T
    value_proj = model.encoder.layer[-1].attention.self.value.weight.T
    Q = torch.matmul(last_layer_hidden, query_proj)
    K = torch.matmul(last_layer_hidden, key_proj)
    V = torch.matmul(last_layer_hidden, value_proj)
    
    # Compute Head Output: attn @ V
    attn_weights_last_layer = attention_weights[-1]
    head_output = torch.matmul(attn_weights_last_layer, V.unsqueeze(1))

    # Combine all components into a dictionary
    components = {
        "Q": Q,
        "K": K,
        "V": V,
        "attention_weights": attention_weights,
        "head_output": head_output,
        "final_output": final_output,
        "hidden_states": hidden_states
    }
    
    # Save each component as a separate JSON file
    for component_name, matrix in components.items():
        file_path = f"{save_directory}\\{output_prefix}_{component_name}.json"
        with open(file_path, "w") as f:
            # Serialize and save each component
            serialized_matrix = serialize_component(matrix)
            json.dump(serialized_matrix, f)
        print(f"Saved {component_name} matrix to {file_path}.")

# Extract Features for Fine-Tuned BERT
extract_bert_features_during_operation(fine_tuned_model, inputs, "fine_tuned_bert")

# Extract Features for BERT Base Uncased
extract_bert_features_during_operation(bert_base_model, inputs, "bert_base_uncased")
