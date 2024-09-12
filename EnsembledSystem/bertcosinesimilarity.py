import torch
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

# Load custom BERT model and tokenizer for embeddings
bert_tokenizer = BertTokenizer.from_pretrained(r'C:\Users\Michalis\Desktop\ceid\HugginFace\final_model_bert')
bert_model = BertModel.from_pretrained(r'C:\Users\Michalis\Desktop\ceid\HugginFace\final_model_bert')

# Load custom BART model and tokenizer for paraphrase generation
bart_tokenizer = BartTokenizer.from_pretrained(r'C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450')
bart_model = BartForConditionalGeneration.from_pretrained(r'C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450')

# Function to generate a paraphrase using BART
def generate_paraphrase_with_bart(sentence):
    inputs = bart_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    
    # Generate paraphrase with BART
    generated_ids = bart_model.generate(
        inputs['input_ids'], 
        max_new_tokens=50, 
        num_beams=5, 
        temperature=0.8, 
        top_k=50, 
        top_p=0.9, 
        early_stopping=True
    )
    
    # Decode the paraphrased output
    paraphrase = bart_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return paraphrase

# Function to calculate BLEU score
def calculate_bleu(reference, hypothesis):
    reference = [nltk.word_tokenize(reference)]
    hypothesis = nltk.word_tokenize(hypothesis)
    return sentence_bleu(reference, hypothesis)

# Function to calculate ROUGE score
def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

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

# Function to evaluate the paraphrase
def evaluate_paraphrase(original_text, paraphrase):
    bleu_score = calculate_bleu(original_text, paraphrase)
    rouge_scores = calculate_rouge(original_text, paraphrase)
    bert_cosine_sim = calculate_bert_cosine_similarity(original_text, paraphrase)

    print(f"\nEvaluation Metrics:")
    print(f"BLEU score: {bleu_score}")
    print(f"ROUGE scores: {rouge_scores}")
    print(f"Cosine Similarity (BERT-based): {bert_cosine_sim}")

# Main function to get input, paraphrase, and evaluate
def main():
    # Get input text from the user
    original_text = input("Enter the text to paraphrase: ")

    # Generate paraphrase with BART
    paraphrase = generate_paraphrase_with_bart(original_text)
    
    # Output the paraphrase
    print(f"\nOriginal text: {original_text}")
    print(f"Paraphrased text: {paraphrase}")
    
    # Evaluate the quality of the paraphrase
    evaluate_paraphrase(original_text, paraphrase)

if __name__ == "__main__":
    main()
