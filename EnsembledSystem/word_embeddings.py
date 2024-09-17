import torch
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration
from bert_score import score  # For BERTScore metric
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import pandas as pd

# Load custom BERT model and tokenizer for embeddings
bert_tokenizer = BertTokenizer.from_pretrained(r'C:\Users\Michalis\Desktop\ceid\HugginFace\final_model_bert')
bert_model = BertModel.from_pretrained(r'C:\Users\Michalis\Desktop\ceid\HugginFace\final_model_bert')

# Load custom BART model and tokenizer for paraphrase generation
bart_tokenizer = BartTokenizer.from_pretrained(r'C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450')
bart_model = BartForConditionalGeneration.from_pretrained(r'C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450')

# Load the MRPC test dataset
def load_mrpc_test_dataset():
    dataset = load_dataset("glue", "mrpc", split='test')  # Load the MRPC test set
    return dataset

# Function to generate a paraphrase using BART
def generate_paraphrase_with_bart(sentence):
    inputs = bart_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    
    # Generate paraphrase with BART
    generated_ids = bart_model.generate(
        inputs['input_ids'], 
        max_new_tokens=50, 
        num_beams=3, 
        temperature=0.8, 
        top_k=50, 
        top_p=0.9, 
        early_stopping=True
    )
    
    # Decode the paraphrased output
    paraphrase = bart_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return paraphrase

# Function to calculate cosine similarity using BERT embeddings
def calculate_bert_cosine_similarity(original_text, paraphrase):
    # Tokenize and encode original and paraphrase using BERT
    original_inputs = bert_tokenizer(original_text, return_tensors="pt", padding=True, truncation=True)
    paraphrase_inputs = bert_tokenizer(paraphrase, return_tensors="pt", padding=True, truncation=True)
    
    # Generate embeddings for original and paraphrase text using BERT
    with torch.no_grad():
        original_embeddings = bert_model(**original_inputs).last_hidden_state.mean(dim=1)  # Average token embeddings
        paraphrase_embeddings = bert_model(**paraphrase_inputs).last_hidden_state.mean(dim=1)  # Average token embeddings
    
    # Compute cosine similarity between the two embeddings
    cosine_sim = cosine_similarity(original_embeddings, paraphrase_embeddings)
    
    return cosine_sim[0][0]  # Return the cosine similarity score

# Function to evaluate the paraphrase using BERTScore and cosine similarity
def evaluate_paraphrase(original_text, paraphrase, reference_text):
    # Calculate BERTScore
    P, R, F1 = score([paraphrase], [reference_text], lang="en", model_type='bert-base-uncased')

    # Calculate cosine similarity between paraphrase and reference
    bert_cosine_sim = calculate_bert_cosine_similarity(reference_text, paraphrase)

    # Create evaluation dictionary
    evaluation_metrics = {
        "bert_score_precision": P.mean().item(),
        "bert_score_recall": R.mean().item(),
        "bert_score_f1": F1.mean().item(),
        "cosine_similarity": bert_cosine_sim
    }
    
    return evaluation_metrics

# Main function to paraphrase and evaluate the MRPC test dataset
def process_mrpc_test_dataset():
    dataset = load_mrpc_test_dataset()
    
    results = []
    
    for i, data in enumerate(dataset):
        original_text = data['sentence1']
        reference_text = data['sentence2']
        
        # Generate paraphrase for the first sentence
        paraphrase = generate_paraphrase_with_bart(original_text)
        
        # Evaluate the paraphrase against the reference (second sentence)
        metrics = evaluate_paraphrase(original_text, paraphrase, reference_text)
        
        # Store the result
        result = {
            "index": i,
            "original_text": original_text,
            "reference_text": reference_text,
            "paraphrase": paraphrase,
            **metrics
        }
        results.append(result)
        
        # Output progress
        if i % 10 == 0:
            print(f"Processed {i} samples")

    # Convert results to DataFrame for analysis
    df_results = pd.DataFrame(results)
    df_results.to_csv("mrpc_paraphrase_test_results.csv", index=False)
    print("Results saved to mrpc_paraphrase_test_results.csv")

if __name__ == "__main__":
    process_mrpc_test_dataset()

