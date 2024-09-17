import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("mrpc_paraphrase_test_results.csv")

# Step 1: Calculate average metrics
average_precision = df['bert_score_precision'].mean()
average_recall = df['bert_score_recall'].mean()
average_f1 = df['bert_score_f1'].mean()
average_cosine_similarity = df['cosine_similarity'].mean()

# Print the average metrics
print(f"Average BERTScore Precision: {average_precision:.4f}")
print(f"Average BERTScore Recall: {average_recall:.4f}")
print(f"Average BERTScore F1: {average_f1:.4f}")
print(f"Average Cosine Similarity: {average_cosine_similarity:.4f}")

# Step 2: Plot the distribution of BERTScore F1 and Cosine Similarity
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(df['bert_score_f1'], bins=20, color='blue', alpha=0.7)
plt.title("BERTScore F1 Distribution")
plt.xlabel("BERTScore F1")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(df['cosine_similarity'], bins=20, color='green', alpha=0.7)
plt.title("Cosine Similarity Distribution")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Step 3: Extract top 5 paraphrases by BERTScore F1 and bottom 5 by Cosine Similarity
top_5_f1 = df.nlargest(5, 'bert_score_f1')
bottom_5_cosine = df.nsmallest(5, 'cosine_similarity')

# Print the top 5 by BERTScore F1
print("\nTop 5 paraphrases by BERTScore F1:")
print(top_5_f1[['original_text', 'paraphrase', 'reference_text', 'bert_score_f1']])

# Print the bottom 5 by Cosine Similarity
print("\nBottom 5 paraphrases by Cosine Similarity:")
print(bottom_5_cosine[['original_text', 'paraphrase', 'reference_text', 'cosine_similarity']])

# Step 4: Optionally save the top and bottom results to separate CSV files
top_5_f1.to_csv("top_5_f1_results.csv", index=False)
bottom_5_cosine.to_csv("bottom_5_cosine_results.csv", index=False)
