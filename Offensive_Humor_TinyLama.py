from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import os

device = "cuda"
print(f"Using device: {device}")

# Step 1: Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token  # Use EOS as PAD

# Step 2: Load and Prepare Dataset
def get_offensive_humor_data(split='train'):
    data = load_dataset('metaeval/offensive-humor', split=split)
    data = data.filter(lambda x: x['title'] is not None and x['selftext'] is not None)
    data = data.map(lambda x: {'prompt': x['title'], 'completion': x['selftext']})
    return data

train_data = get_offensive_humor_data('train')

def tokenize_function(examples):
    prompts = [p if p else "" for p in examples['prompt']]
    completions = [c if c else "" for c in examples['completion']]
    
    inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=128)
    outputs = tokenizer(completions, padding="max_length", truncation=True, max_length=128)
    
    inputs['labels'] = outputs['input_ids']
    return inputs

train_data = train_data.map(tokenize_function, batched=True)

print(train_data[0])

print("Downloading TinyLlama Model....")

# Load Model WITHOUT Quantization
tinyllama_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
).to(device)

# Step 3: Attach LoRA Adapters
lora_config = LoraConfig(
    r=8,  # Reduced for stability
    lora_alpha=16,  # Adjusted scaling factor
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.05,  # Lower dropout for better balance
    bias="none",
    task_type="CAUSAL_LM"
)

tinyllama_model = get_peft_model(tinyllama_model, lora_config)
tinyllama_model.print_trainable_parameters()  # Ensure LoRA is applied

# Step 4: Configure Training
training_args = TrainingArguments(
    output_dir="./Offensive_Humor_TL",
    num_train_epochs=1,  
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,  
    logging_dir='./Offensive_Humor_Logg',
    save_strategy="steps",
    save_steps=1000,
    logging_steps=10,
    fp16=True,
    warmup_steps=2000, 
    learning_rate=2e-4,  
    lr_scheduler_type="cosine",  
)

# Step 5: Train Model
trainer = Trainer(
    model=tinyllama_model,
    args=training_args,
    train_dataset=train_data,
)

checkpoint_path = None
if os.path.exists("./Offensive_Humor_TL"):
    checkpoints = [ckpt for ckpt in os.listdir("./Offensive_Humor_TL") if ckpt.startswith("checkpoint")]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        checkpoint_path = os.path.join("./Offensive_Humor_TL", latest_checkpoint)
        print(f"Resuming from checkpoint: {checkpoint_path}")

try:
    trainer.train(resume_from_checkpoint=checkpoint_path)
except Exception as e:
    print(f"Training interrupted due to: {e}")
    trainer.save_model("./fine_tuned_model_OF_DeepSeek")  # Save current progress

# Step 6: Save Fine-Tuned LoRA Model
trainer.model.save_pretrained("./fine_tuned_model_OF_TL")

print("Fine-tuning completed successfully ✅")

tokenizer.save_pretrained("./fine_tuned_model_OF_TL")

print("Tokenizer successfully Saved ✅")
