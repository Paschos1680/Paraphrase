


# Import necessary libraries
from transformers import BartForConditionalGeneration, BartTokenizer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from datasets import load_dataset, concatenate_datasets
import torch

def load_and_preprocess_datasets():
    # Load datasets
    paws = load_dataset("paws", "labeled_final", split='train')
    quora = load_dataset("quora", split='train')
    mrpc = load_dataset("glue", "mrpc", split='train').remove_columns(['label', 'idx'])

    # Combine datasets
    combined_dataset = concatenate_datasets([paws, quora, mrpc])

    # Initialize tokenizer
    tokenizer = BartTokenizer.from_pretrained("eugenesiow/bart-paraphrase")

    # Define preprocessing function to handle both inputs and labels
    def preprocess_function(examples):
        inputs = tokenizer(["paraphrase: " + (doc if doc is not None else "") for doc in examples['sentence1']],
                           max_length=128, truncation=True, padding="max_length", return_tensors="pt")
        labels = tokenizer([doc if doc is not None else "" for doc in examples['sentence2']],
                           max_length=128, truncation=True, padding="max_length", return_tensors="pt")

        # Return data as lists to ensure compatibility with `datasets` batched mapping
        return {'input_ids': inputs['input_ids'].tolist(), 'attention_mask': inputs['attention_mask'].tolist(), 'labels': labels['input_ids'].tolist()}

    tokenized_datasets = combined_dataset.map(preprocess_function, batched=True, remove_columns=combined_dataset.column_names)
    return tokenized_datasets, tokenizer

try:
    tokenized_datasets, tokenizer = load_and_preprocess_datasets()
    model = BartForConditionalGeneration.from_pretrained("eugenesiow/bart-paraphrase")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./eugenesiow_bart_final",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        logging_dir='./logs'
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained("./eugenesiow_bart_final")
    tokenizer.save_pretrained("./eugenesiow_bart_final")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    # Save model and tokenizer periodically or upon an exception
    model.save_pretrained("./eugenesiow_bart_final_backup")
    tokenizer.save_pretrained("./eugenesiow_bart_final_backup")
    raise
