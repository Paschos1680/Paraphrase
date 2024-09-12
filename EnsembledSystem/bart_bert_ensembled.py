import torch
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk

# Load custom BERT model and tokenizer for embeddings (BERT for text analysis, optional)
bert_tokenizer = BertTokenizer.from_pretrained(r'C:\Users\Michalis\Desktop\ceid\HugginFace\final_model_bert')
bert_model = BertModel.from_pretrained(r'C:\Users\Michalis\Desktop\ceid\HugginFace\final_model_bert')

# Load custom BART model and tokenizer for paraphrase generation (BART for generation)
bart_tokenizer = BartTokenizer.from_pretrained(r'C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450')
bart_model = BartForConditionalGeneration.from_pretrained(r'C:\Users\Michalis\Desktop\ceid\HugginFace\FinalBART\checkpoint-7450')

# Function to analyze text using BERT (optional)
def analyze_with_bert(sentence):
    inputs = bert_tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state
    print(f"Generated BERT embeddings: {embeddings.shape}")
    return embeddings

# Function to generate a paraphrase using BART with adjusted generation parameters
def generate_paraphrase_with_bart(sentence):
    inputs = bart_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    
    # Adjusted generation parameters for BART
    generated_ids = bart_model.generate(
        inputs['input_ids'], 
        max_new_tokens=50,     # Adjust the token length
        num_beams=5,           # Increase beam search for more diverse exploration
        temperature=0.8,       # Control diversity (higher = more diverse)
        top_k=50,              # Top-k sampling (limit to top 50 tokens)
        top_p=0.8,             # Nucleus sampling (consider tokens with 90% cumulative probability)
        early_stopping=True    # Stop when a complete sentence is generated
    )
    
    # Decode the generated token IDs to text
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

# Function to calculate cosine similarity
def calculate_cosine_similarity(reference, hypothesis):
    vectorizer = TfidfVectorizer().fit_transform([reference, hypothesis])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

# Function to evaluate the paraphrase
def evaluate_paraphrase(original_text, paraphrase):
    bleu_score = calculate_bleu(original_text, paraphrase)
    rouge_scores = calculate_rouge(original_text, paraphrase)
    cosine_sim = calculate_cosine_similarity(original_text, paraphrase)

    print(f"\nEvaluation Metrics:")
    print(f"BLEU score: {bleu_score}")
    print(f"ROUGE scores: {rouge_scores}")
    print(f"Cosine Similarity: {cosine_sim}")

# Main function to get input, paraphrase, and evaluate
def main():
    # Get input text from the user
    original_text = input("Enter the text to paraphrase: ")

    # Analyze with BERT (optional, can be skipped if not needed)
    bert_embeddings = analyze_with_bert(original_text)
    
    # Generate paraphrase with BART
    paraphrase = generate_paraphrase_with_bart(original_text)
    
    # Output the paraphrase
    print(f"\nOriginal text: {original_text}")
    print(f"Paraphrased text: {paraphrase}")
    
    # Evaluate the quality of the paraphrase
    evaluate_paraphrase(original_text, paraphrase)

if __name__ == "__main__":
    main()



