# Install necessary libraries
!pip install transformers datasets torch scikit-learn scipy accelerate

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, ClassLabel, Value
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Load tokenizer and model
model_name = 'Prompsit/paraphrase-bert-en'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load datasets
datasets = {
    "mrpc": load_dataset("glue", "mrpc", split='train'),
    "paws": load_dataset("paws", "labeled_final", split='train'),
    "sts": load_dataset("stsb_multi_mt", name="en", split='train')
}

# Inspect datasets to verify structure
for name, dataset in datasets.items():
    print(f"Inspecting {name} dataset columns and first example:")
    print(dataset.column_names)
    print(dataset.features)
    print(dataset[0])

# Function to preprocess datasets
def preprocess_function(examples, dataset_name):
    if dataset_name == "mrpc" or dataset_name == "paws":
        inputs = tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = examples["labels"]
    elif dataset_name == "sts":
        inputs = tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = [1 if score > 2.5 else 0 for score in examples["similarity_score"]]
    return inputs

# Ensure all datasets have the same label column name and type before preprocessing
for name, dataset in datasets.items():
    if 'label' in dataset.column_names:
        dataset = dataset.rename_column('label', 'labels')
    datasets[name] = dataset

# Check and convert ClassLabel to int if needed
for name, dataset in datasets.items():
    if 'labels' in dataset.column_names and isinstance(dataset.features['labels'], ClassLabel):
        dataset = dataset.cast_column('labels', Value('int64'))
    datasets[name] = dataset

# Preprocess and format datasets
for name, dataset in datasets.items():
    print(f"Preprocessing {name} dataset...")
    def preprocess_wrapper(examples):
        if 'labels' in examples:
            return preprocess_function(examples, name)
        else:
            return preprocess_function(examples, name)
    
    dataset = dataset.map(preprocess_wrapper, batched=True, remove_columns=[col for col in dataset.column_names if col not in ['sentence1', 'sentence2']])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    datasets[name] = dataset

# Concatenate datasets and split into train and test
combined_dataset = concatenate_datasets([ds for ds in datasets.values()])
combined_dataset = combined_dataset.train_test_split(test_size=0.1)
train_dataset = combined_dataset['train']
test_dataset = combined_dataset['test']

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=10,
)

# Learning rate scheduler and optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * training_args.num_train_epochs)

# Fine-tuning strategy: Gradual unfreezing
for param in model.bert.embeddings.parameters():
    param.requires_grad = False

for layer in model.bert.encoder.layer[:6]:
    for param in layer.parameters():
        param.requires_grad = False

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    optimizers=(optimizer, scheduler)
)

# Train the model with frozen layers
trainer.train()

# Unfreeze additional layers and continue training
for layer in model.bert.encoder.layer[:6]:
    for param in layer.parameters():
        param.requires_grad = True

trainer.train()

# Save the model
model.save_pretrained('./final_bert_paraphrase')
tokenizer.save_pretrained('./final_bert_paraphrase')





