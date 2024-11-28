import json
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset

# Paths to your models
pretrained_model_name = "facebook/bart-large"
fine_tuned_model_path = "C:/Users/Michalis/Desktop/ceid/HugginFace/FinalBART/checkpoint-7450"

# Directory to save attention matrices
save_directory = "C:/Users/Michalis/Desktop/ceid/HugginFace/Github/ExplainingBart/mm"

# Load tokenizer and models
tokenizer = BartTokenizer.from_pretrained(pretrained_model_name)
pretrained_model = BartForConditionalGeneration.from_pretrained(pretrained_model_name, output_attentions=True)
fine_tuned_model = BartForConditionalGeneration.from_pretrained(fine_tuned_model_path, output_attentions=True)

# Load MRPC dataset and get the first sentence
mrpc_test = load_dataset("glue", "mrpc", split="test")
first_sentence = mrpc_test[0]["sentence1"]

# Function to extract and save attention matrices for specific layers
def extract_attention_for_layers(model, tokenizer, sentence, output_prefix):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        output_attentions=True,
        return_dict_in_generate=True
    )

    # Attention matrices for first, middle, and last layers
    layers = {
        "first_layer": outputs.decoder_attentions[0][0].detach().numpy(),  # First decoder layer
        "middle_layer": outputs.decoder_attentions[len(outputs.decoder_attentions) // 2][0].detach().numpy(),  # Middle layer
        "last_layer": outputs.decoder_attentions[-1][0].detach().numpy()  # Last decoder layer
    }

    # Save attention matrices for all layers
    for layer_name, attention_matrix in layers.items():
        output_file = f"{save_directory}/{output_prefix}_{layer_name}.json"
        attention_data = {"matrix": attention_matrix.tolist()}
        with open(output_file, "w") as f:
            json.dump(attention_data, f)
        print(f"Saved attention matrix for {layer_name} to {output_file}")

# Extract attention matrices for both models
extract_attention_for_layers(pretrained_model, tokenizer, first_sentence, "pretrained")
extract_attention_for_layers(fine_tuned_model, tokenizer, first_sentence, "fine_tuned")
