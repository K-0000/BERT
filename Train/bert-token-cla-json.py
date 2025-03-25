from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import pandas as pd
import json

# Load and preprocess data
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return pd.DataFrame(json.load(f))

df = load_json_data("maindataset.json")

# Label configuration
label_list = ['O', 'ADDRESS', 'ITEM', 'QUANTITY', 'PHONE']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Convert labels to IDs
df['token_labels'] = df['token_labels'].apply(
    lambda labels: [label2id[label] for label in labels]
)

# Validate and filter dataset
dataset = Dataset.from_pandas(df).filter(
    lambda x: len(x['tokens']) == len(x['token_labels'])
)

# Tokenizer with dynamic padding
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize_and_align(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=False,  # Will handle padding in collator
        is_split_into_words=True,
        return_overflowing_tokens=True
    )
    
    labels = []
    for i, label in enumerate(examples["token_labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(label[word_idx])
                except IndexError:
                    label_ids.append(-100)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        # Align label length with input_ids
        label_ids = label_ids[:len(tokenized_inputs["input_ids"][i])]
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(
    tokenize_and_align,
    batched=True,
    remove_columns=dataset.column_names,
    batch_size=32
)

# Data collator for dynamic padding
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding='longest',
    label_pad_token_id=-100
)

# Split dataset
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# Model initialization
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Reduced for stability
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    gradient_accumulation_steps=2
)

# Trainer with proper initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Start training
print("Starting training...")
trainer.train()

# Save model
model.save_pretrained("./my_ner_model")
tokenizer.save_pretrained("./my_ner_model")
print("Training complete!")