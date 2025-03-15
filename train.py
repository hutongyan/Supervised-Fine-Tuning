import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from nltk.translate.bleu_score import sentence_bleu

# ✅ Use GPU 0 and GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["NCCL_P2P_DISABLE"] = "1"  # Avoid NCCL communication errors

# ✅ Device binding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using {torch.cuda.device_count()} GPUs for training!")

# === Configure Paths ===
MODEL_PATH = "/home/guest/zsf/hty/models/models--EleutherAI--pythia-160m/snapshots/50f5173d932e8e61f858120bcb800b97af589f46"
DATASET_PATH = "/home/guest/zsf/hty/sft/mistral_data"
OUTPUT_DIR = "/home/guest/zsf/hty/sft/output"
LOG_DIR = "/home/guest/zsf/hty/sft/logs"
DEEPSPEED_CONFIG = "/home/guest/zsf/hty/sft/ds_config.json"

EPOCHS = 3
BATCH_SIZE = 32  
LEARNING_RATE = 2e-5
MAX_LENGTH = 2048
FP16 = torch.cuda.is_available()

# === Load Dataset ===
print("Loading dataset...")
dataset = load_from_disk(DATASET_PATH)
print("Checking dataset columns:", dataset["test"].column_names)

# === Load Model and Tokenizer ===
print("Loading model & tokenizer from local path...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16
)
model.config.use_cache = False  # ✅ Disable `use_cache` to avoid DeepSpeed conflicts
model.to(device)  

# ✅ Enable gradient checkpointing to reduce memory usage
model.gradient_checkpointing_enable()

# === Preprocessing Data ===
def preprocess_function(examples):
    """Tokenize and retain `formatted_text`"""
    texts = examples.get("formatted_text", [""])
    new_texts = [" ".join(t) if isinstance(t, list) else str(t) for t in texts]

    # ✅ Perform tokenization
    tokenized = tokenizer(new_texts, padding="max_length", truncation=True, max_length=MAX_LENGTH)

    # ✅ Use `original_text` only for evaluation
    tokenized["original_text"] = new_texts
    return tokenized

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

# === Training Parameters ===
print("Setting training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    logging_dir=LOG_DIR,
    logging_steps=500,
    save_total_limit=2,
    report_to="none",
    fp16=FP16,
    deepspeed=DEEPSPEED_CONFIG,
    remove_unused_columns=False
)

# === Trainer Initialization ===
print("Initializing Trainer...")
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Start Training ===
print("🚀 Starting training...")
trainer.train()

# === Evaluate BLEU Score ===
print("Evaluating model...")
bleu_scores = []
for i, example in enumerate(dataset["test"]):
    reference_text = example.get("formatted_text", "").strip()
    if not reference_text:
        print(f"Skipping invalid reference at index {i}")
        continue

    input_ids = tokenizer(reference_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).input_ids.to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)  # ✅ Generate attention_mask
    hypothesis_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,  # ✅ Solve attention_mask warning
        max_new_tokens=256,  # ✅ Control generated length
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id  # ✅ Avoid `pad_token_id` warning
    )
    hypothesis_text = tokenizer.decode(hypothesis_ids[0].tolist(), skip_special_tokens=True)

    reference = reference_text.split()
    hypothesis = hypothesis_text.split()

    if len(reference) > 0 and len(hypothesis) > 0:
        score = sentence_bleu([reference], hypothesis)
        bleu_scores.append(score)

avg_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
print(f"✅ Average BLEU Score: {avg_bleu_score:.4f}")

# === Save Model ===
print("Saving fine-tuned model locally...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
