# Import necessary libraries
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from captum.attr import LayerConductance
import matplotlib.pyplot as plt

# Paths for models
model_path_fine_tuned = r"C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450"
model_path_pretrained = "facebook/bart-large"

# Load pretrained model and tokenizer for Facebook BART
pretrained_model = BartForConditionalGeneration.from_pretrained(model_path_pretrained)
tokenizer = BartTokenizer.from_pretrained(model_path_pretrained)

# Load fine-tuned BART model
fine_tuned_model = BartForConditionalGeneration.from_pretrained(model_path_fine_tuned)

# Move models to CPU (remove .to("cuda"))
pretrained_model.to("cpu")
fine_tuned_model.to("cpu")

# Example sentence for testing TCAV
examples = [
    "The cat sat on the mat.",
    "It was raining heavily outside.",
    "He kicked the ball towards the goal.",
]

# Tokenize inputs
inputs = tokenizer(examples, return_tensors="pt", padding=True, truncation=True)

# Define concepts for TCAV
concepts = ["weather", "action", "idiom"]

# Placeholder for TCAV computation
def run_tcav(model, inputs, concepts):
    # Run the TCAV process here
    for idx, concept in enumerate(concepts):
        print(f"Running TCAV for concept '{concept}' using example sentence {idx + 1}:")
        # Example of printing TCAV without actual computation code for now
        print(f"TCAV score for concept '{concept}' on example {idx + 1}: [Placeholder]")

# Run TCAV on both models for comparison
print("Running TCAV on Pretrained Model")
run_tcav(pretrained_model, inputs, concepts)

print("\nRunning TCAV on Fine-Tuned Model")
run_tcav(fine_tuned_model, inputs, concepts)

# Visualize placeholders
plt.figure(figsize=(8, 4))
for i, concept in enumerate(concepts):
    plt.bar(i, [0.8, 0.6, 0.7][i], label=concept)  # Placeholder values
plt.legend()
plt.xlabel("Concepts")
plt.ylabel("TCAV Score")
plt.title("Placeholder TCAV Scores for Concepts")
plt.show()

