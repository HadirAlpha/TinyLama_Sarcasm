from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import os

device = "cuda"
print(f"Using device: {device}")

# Step 1: Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-step-50K-105b")
tokenizer.pad_token = tokenizer.eos_token  # Use EOS as PAD

# Step 2: Load and Prepare Dataset
def get_sarcasm_data(split='train'):
    data = load_dataset('daniel2588/sarcasm', split=split)
    data = data.map(lambda x: {'prompt': x['parent_comment'], 'completion': x['comment']})
    return data

train_data = get_sarcasm_data('train')

def tokenize_function(examples):
    prompts = [str(p) if p is not None else "" for p in examples['prompt']]
    completions = [str(c) if c is not None else "" for c in examples['completion']]
    
    inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=64)
    outputs = tokenizer(completions, padding="max_length", truncation=True, max_length=64)
    
    inputs['labels'] = outputs['input_ids']
    return inputs

train_data = train_data.map(tokenize_function, batched=True)

print("Downloading TinyLlama Model....")

# ✅ Load Model WITHOUT Quantization
tinyllama_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-step-50K-105b"
).to(device)

# ✅ Step 3: Attach LoRA Adapters
lora_config = LoraConfig(
    r=32,  
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

tinyllama_model = get_peft_model(tinyllama_model, lora_config)
tinyllama_model.print_trainable_parameters()  # Ensure LoRA is applied

# Step 4: Configure Training
training_args = TrainingArguments(
    output_dir="./res",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_dir='./Logs',
    save_strategy="steps",
    save_steps=1000,
    logging_steps=10,
    fp16=True,
    warmup_steps=1000, 
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
)

# Step 5: Train Model
trainer = Trainer(
    model=tinyllama_model,
    args=training_args,
    train_dataset=train_data,
)

checkpoint_path = None
if os.path.exists("./res"):
    checkpoints = [ckpt for ckpt in os.listdir("./res") if ckpt.startswith("checkpoint")]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        checkpoint_path = os.path.join("./res", latest_checkpoint)
        print(f"Resuming from checkpoint: {checkpoint_path}")

try:
    trainer.train(resume_from_checkpoint=checkpoint_path)
except Exception as e:
    print(f"Training interrupted due to: {e}")
    trainer.save_model("./Fine_Tuned_Model")  # Save current progress

# Step 6: Save Fine-Tuned LoRA Model
trainer.model.save_pretrained("./Fine_Tuned_Model")

print("✅ Fine-tuning completed successfully!")

tokenizer.save_pretrained("./Fine_Tuned_Model")

print("✅ Tokenizer successfully!")
