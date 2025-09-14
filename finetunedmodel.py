

# pip install datasets transformers torch

import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    pipeline
)
import os
import sys

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Read the text file (Vast.ai path)
text_file_path = "/root/cnn_dailymail_full.txt"  # Vast.ai path
if not os.path.exists(text_file_path):
    print(f"Error: File {text_file_path} not found!")
    print("Please upload your dataset file to the instance.")
    sys.exit(1)

print(f"Reading dataset from: {text_file_path}")
with open(text_file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Raw text length: {len(raw_text):,} characters")

# DATASET TRUNCATION FOR MANAGEABLE TRAINING
max_chars = 50000000  # 50MB limit for manageable training
if len(raw_text) > max_chars:
    print(f" Dataset too large ({len(raw_text):,} chars). Truncating to {max_chars:,} chars...")
    raw_text = raw_text[:max_chars]
    print(f" Truncated to {len(raw_text):,} characters ({len(raw_text)/1e6:.1f}MB)")
else:
    print(f" Dataset size is manageable: {len(raw_text):,} characters ({len(raw_text)/1e6:.1f}MB)")

# Initialize tokenizer
print("Loading GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the entire text
print("Tokenizing text...")
tokens = tokenizer(raw_text, return_tensors="pt")
input_ids = tokens["input_ids"][0]

print(f"Total tokens: {len(input_ids):,}")

# Create training examples with block size
block_size = 128
examples = []
for i in range(0, len(input_ids) - block_size, block_size):
    examples.append(input_ids[i:i+block_size])

print(f"Created {len(examples):,} training examples")

# LIMIT TRAINING EXAMPLES FOR FASTER TRAINING
max_examples = 10000  # Limit to 10K examples for faster training
if len(examples) > max_examples:
    print(f"  Too many examples ({len(examples):,}). Limiting to {max_examples:,} examples...")
    examples = examples[:max_examples]
    print(f" Limited to {len(examples):,} training examples")
else:
    print(f"Using all {len(examples):,} training examples")

class DialogueDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx],
            "labels": self.examples[idx]
        }

# Split into train and validation
split_idx = int(0.9 * len(examples))
train_dataset = DialogueDataset(examples[:split_idx])
val_dataset = DialogueDataset(examples[split_idx:])

print(f"Train examples: {len(train_dataset):,}")
print(f"Validation examples: {len(val_dataset):,}")

# Initialize model
print("Loading GPT-2 model...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)
print(f"Model loaded on {device}")

# OPTIMAL TRAINING ARGUMENTS FOR VAST.AI
training_args = TrainingArguments(
    # Output and logging
    output_dir="/root/creative_results",  # Vast.ai path
    overwrite_output_dir=True,
    logging_dir="/root/creative_logs",    # Vast.ai path
    logging_steps=25,  # More frequent logging
    
    # Training parameters optimized for Vast.ai
    num_train_epochs=5,  # More epochs since dataset is smaller
    per_device_train_batch_size=4,  # Larger batch size possible
    per_device_eval_batch_size=4,
    
    # Learning rate optimization
    learning_rate=5e-5,  # Slightly higher for smaller dataset
    warmup_steps=100,  # Shorter warmup
    weight_decay=0.01,  # Regularization to prevent overfitting
    
    # BUILT-IN VALIDATION MONITORING
    evaluation_strategy="steps",  # Evaluate more frequently
    eval_steps=200,  # Evaluate every 200 steps (more frequent)
    save_strategy="steps",
    save_steps=500,  # Save more frequently
    save_total_limit=3,  # Keep more checkpoints
    
    # Model selection with early stopping
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Memory optimizations for Vast.ai GPU
    gradient_accumulation_steps=2,  # Effective batch size = 4 * 2 = 8
    fp16=True,  # Use mixed precision for speed and memory
    dataloader_num_workers=2,  # Faster data loading
    remove_unused_columns=False,  # Keep all data
    prediction_loss_only=True,  # Focus on language modeling loss
    
    # Reporting
    report_to="none",
    
    # Built-in logging for monitoring
    logging_first_step=True,  # Log first step
    logging_strategy="steps",  # Log every 25 steps
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
print("Starting training on Vast.ai...")
print("=" * 60)
print(" Vast.ai Optimized Training:")
print("   - Dataset truncated to 50MB")
print("   - Limited to 10K training examples")
print("   - Optimized for RTX 3090/3080")
print("   - Training loss every 25 steps")
print("   - Validation loss every 200 steps")
print("   - Results saved to /root/creative_results")
print("=" * 60)

trainer.train()

# Save the model
model_save_path = "/root/creative-finetuned-model"  # Vast.ai path
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Creative model saved to {model_save_path}")

# Evaluate the model
eval_result = trainer.evaluate()
print(f"Final Validation loss: {eval_result['eval_loss']:.4f}")

print("Training completed successfully on Vast.ai!")

# TEXT GENERATION USING HUGGINGFACE PIPELINE
print("\n" + "="*60)
print(" TEXT GENERATION WITH HUGGINGFACE PIPELINE")
print("=" * 60)

# Create text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Test prompts relevant to CNN news dataset
test_prompts = [
    "WASHINGTON (CNN) --",
    "LONDON, England (Reuters) --",
    "MINNEAPOLIS, Minnesota (CNN) --",
    "The president said",
    "According to officials",
    "In a statement released",
    "The investigation revealed",
    "Witnesses reported that",
    "The incident occurred",
    "Police confirmed that"
]

print("\n Testing CNN news-style prompts:")
print("-" * 50)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n Test {i}: '{prompt}'")
    
    # Generate with different parameters
    result = text_generator(
        prompt,
        max_length=80,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1
    )
    
    generated_text = result[0]['generated_text']
    print(f" Generated: '{generated_text}'")
    print("-" * 30)

# Test different creativity levels with news-style prompts
print("\n Testing different creativity levels for news generation:")
print("-" * 50)

prompt = "WASHINGTON (CNN) -- Officials announced"

# High creativity
print(f"\n High Creativity (temperature=1.2):")
result = text_generator(prompt, max_length=100, temperature=1.2, top_k=100)
print(f" Generated: '{result[0]['generated_text']}'")

# Balanced creativity
print(f"\n  Balanced Creativity (temperature=0.8):")
result = text_generator(prompt, max_length=100, temperature=0.8, top_k=50)
print(f"Generated: '{result[0]['generated_text']}'")

# Low creativity
print(f"\n Low Creativity (temperature=0.5):")
result = text_generator(prompt, max_length=100, temperature=0.5, top_k=20)
print(f" Generated: '{result[0]['generated_text']}'")

print("\n Text generation completed!")
print("\n Tips for news-style generation:")
print("   - Use location + source format: 'WASHINGTON (CNN) --'")
print("   - Include attribution: 'officials said', 'witnesses reported'")
print("   - Add quotes and statements for authenticity")
print("   - Include specific details and facts")
print("   - Maintain journalistic tone and style")

print("\n Files saved:")
print(f"   - Model: {model_save_path}")
print(f"   - Logs: /root/creative_logs")

print(f"   - Results: /root/creative_results") 

