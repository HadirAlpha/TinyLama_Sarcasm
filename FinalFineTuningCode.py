from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import os

# Detect if a GPU is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Step 1: Load the tokenizer for the DeepSeek model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Step 2: Load and preprocess the dataset
def get_offensive_humor_data(split='train'):
    data = load_dataset('metaeval/offensive-humor', split=split)
    # Convert dataset columns to match prompt-completion format
    data = data.map(lambda x: {'prompt': x['title'], 'completion': x['selftext']})
    return data

train_data = get_offensive_humor_data('train')

# Tokenize inputs and outputs for model training
def tokenize_function(examples):
    prompts = [p if p else "" for p in examples['prompt']]
    completions = [c if c else "" for c in examples['completion']]
    
    inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(completions, padding="max_length", truncation=True, max_length=512)
    
    inputs['labels'] = outputs['input_ids']
    return inputs

train_data = train_data.map(tokenize_function, batched=True)

print("Sample tokenized data:", train_data[0])

print("Downloading DeepSeek Model....")

# Step 3: Load the DeepSeek model onto the selected device
DeepSeek_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").to(device)

# Step 4: Apply LoRA adapters for efficient fine-tuning
lora_config = LoraConfig(
    r=16,  # Rank parameter for LoRA, reducing trainable parameters
    lora_alpha=32,  # Scaling factor for LoRA updates
    target_modules=["q_proj", "v_proj"],  # Specific layers to apply LoRA
    lora_dropout=0.05,  # Dropout to improve generalization
    bias="none",
    task_type="CAUSAL_LM"
)

DeepSeek_model = get_peft_model(DeepSeek_model, lora_config)
DeepSeek_model.print_trainable_parameters()  # Display number of trainable parameters

# Step 5: Define training configurations
training_args = TrainingArguments(
    output_dir="./Offensive_Humor_DeepSeek",  # Directory for saving model checkpoints
    num_train_epochs=1,  # Number of training epochs
    per_device_train_batch_size=16,  # Batch size per device
    gradient_accumulation_steps=2,  # Accumulate gradients over multiple steps
    logging_dir='./Offensive_Humor_DeepSeek_Log',  # Directory for logs
    save_strategy="steps",  # Save checkpoints at specific intervals
    save_steps=1000,  # Save every 1000 steps
    logging_steps=5,  # Log every 5 steps
    fp16=True,  # Enable mixed precision training for efficiency
    learning_rate=1e-4,  # Learning rate for optimizer
    lr_scheduler_type="cosine",  # Use cosine decay learning rate schedule
)

# Step 6: Initialize the Trainer class and begin training
trainer = Trainer(
    model=DeepSeek_model,
    args=training_args,
    train_dataset=train_data,
)

trainer.train()

# Step 7: Save the fine-tuned model and tokenizer
trainer.model.save_pretrained("./fine_tuned_model_OF_DeepSeek")
print("Fine-tuning completed successfully ✅")

print("Model saved in ./fine_tuned_model_OF_DeepSeek folder ✅")

tokenizer.save_pretrained("./fine_tuned_model_OF_DeepSeek")
print("Tokenizer saved successfully ✅")
