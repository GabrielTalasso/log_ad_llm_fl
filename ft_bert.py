import os
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#load nrows
nrows = None
tokenized_datasets = DatasetDict.load_from_disk('../.dataset/hdfs/tokenized', keep_in_memory=True)

if nrows:
    tokenized_datasets["train"] = tokenized_datasets["train"].select(list(range(nrows)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Step 3: Apply LoRA Configuration
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key"],  # Inject LoRA into these modules
)
model = get_peft_model(model, lora_config)

# Step 5: Mask Tokens for MLM
# Data collator for MLM handles masking
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,  # 15% tokens will be masked
)

# Step 6: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./lora-bert",
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=2e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    per_device_train_batch_size=32,
    save_steps=1000,
    #set validayion
    evaluation_strategy="steps",
    eval_steps=5000,
    fp16=torch.cuda.is_available(),  # Enable mixed precision if using GPU
)

# Step 7: Custom Trainer to Save Losses
class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_losses = []
        self.validation_losses = []

    def log(self, logs):
        super().log(logs)
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.validation_losses.append(logs["eval_loss"])

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 8: Train the Model
trainer.train()

# Save the fine-tuned model and losses
model.save_pretrained("./lora-bert-finetuned")
tokenizer.save_pretrained("./lora-bert-finetuned")
with open("training_losses.json", "w") as f:
    json.dump(trainer.train_losses, f)
with open("validation_losses.json", "w") as f:
    json.dump(trainer.validation_losses, f)

# Step 9: Test the Model
# Define evaluation function
def evaluate_model(model, test_dataset, tokenizer):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in test_dataset:
            inputs = tokenizer(batch["text"], return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            labels = inputs.input_ids.clone()
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1
    return total_loss / num_batches

# Evaluate both pre-trained and fine-tuned models
test_data = tokenized_datasets["test"]
print('Evaluating - Test Data Length:', len(test_data))

pretrained_loss = evaluate_model(AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device), test_data, tokenizer)
finetuned_loss = evaluate_model(model, test_data, tokenizer)

print(f"Pre-trained BERT Test Loss: {pretrained_loss}")
print(f"Fine-tuned BERT Test Loss: {finetuned_loss}")
