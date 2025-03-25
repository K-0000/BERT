import torch
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import json

# Load JSON data
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Load dataset from JSON
df = load_json_data("maindataset.json")

df['label'] = df['label'] - 1

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split dataset
dataset = dataset.train_test_split(test_size=0.2)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256  # Increased from default for Burmese text
    )

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Model initialization
#num_labels = len(df['label'].unique())  # Automatically detect number of classes
#num_labels = df['label'] - 1
#df['label'] = df['label'] - 1
#unique_labels = df['label'].unique()
#print("Unique labels:", unique_labels)  # Debug: check what you get
#num_labels = int(len(unique_labels))
num_labels = 9
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=num_labels
)

# Training arguments with improvements for Burmese text
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Reduced for stability
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,
    fp16=True if torch.cuda.is_available() else False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Start training
print("Starting training...")
trainer.train()

# Evaluate
results = trainer.evaluate()
print(f"Final evaluation results: {results}")

# Save model
model.save_pretrained("./myanmar_bert_text_classification")
tokenizer.save_pretrained("./myanmar_bert_text_classification")
print("Model saved successfully!")

model = BertForSequenceClassification.from_pretrained("./myanmar_bert_text_classification")
tokenizer = BertTokenizer.from_pretrained("./myanmar_bert_text_classification")

inputs = tokenizer("မင်းရဲကျော်စွာလမ်း တပ်ဖွဲ့မှတ်တိုင်နား ဒေါပုံ", return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
print(predictions)

inputs = tokenizer("09420046128", return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
print(predictions)


inputs = tokenizer(" မင်းရဲကျော်စွာလမ်း တပ်ဖွဲ့မှတ်တိုင်နား ဒေါပုံ 09420046128", return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
print(predictions)