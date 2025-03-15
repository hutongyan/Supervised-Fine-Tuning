import os
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

# === 1️⃣ Configure Paths ===
DATASET_PATH = "/home/guest/zsf/hty/sft/deita-6k-v0"
SAVE_PATH = "/home/guest/zsf/hty/sft/mistral_data"
MAX_TOKENS = 2048

# === 2️⃣ Load Dataset ===
print("Loading dataset...")
dataset = load_from_disk(DATASET_PATH)
print(f"✅ Dataset loaded: {len(dataset['train'])} samples")

# === 3️⃣ Initialize Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# === 4️⃣ Mistral-7B-Instruct Chat Template ===
def apply_mistral_chat_template(conversations):
    """Manually implement the Mistral-7B-Instruct chat template."""
    formatted_chat = []
    for i, message in enumerate(conversations):
        role = message["role"]
        content = message["content"]

        if role == "user":
            if i == 0:
                formatted_chat.append(f"<s>[INST] {content} [/INST]")
            else:
                formatted_chat.append(f"\n[INST] {content} [/INST]")
        elif role == "assistant":
            formatted_chat.append(f"{content}</s>")

    return ''.join(formatted_chat).strip()

# === 5️⃣ Format Dataset ===
def format_with_custom_chat_template(example):
    """Process conversations in batches and apply the chat template."""
    formatted_texts = []  # 🚀 Store results for each sample in the batch

    for conversations in example["conversations"]:  # Process each batch
        chat = []
        for message in conversations:
            if isinstance(message, dict) and "from" in message and "value" in message:
                role = "user" if message["from"] == "human" else "assistant"
                chat.append({"role": role, "content": message["value"]})

        if chat:
            formatted_texts.append(apply_mistral_chat_template(chat))
        else:
            formatted_texts.append("")  # Ensure batch length consistency

    return {"formatted_text": formatted_texts}  # ✅ Ensure it returns a list

print("Formatting dataset...")
dataset = dataset.map(format_with_custom_chat_template, batched=True)
print(f"✅ Formatting completed. Sample:\n{dataset['train'][0]['formatted_text'][:500]}")

# === 6️⃣ Split Overlength Text (By Conversation Rounds) ===
def split_conversation_by_rounds(example):
    """Split conversation rounds to ensure each sample is ≤ 2048 tokens."""
    text = example["formatted_text"]
    
    # Split by conversation rounds where each `[INST] ... [/INST]` is a round
    rounds = text.split("[INST]")  
    rounds = ["[INST]" + r for r in rounds if r.strip()]  # Re-add [INST]

    new_samples = []
    current_segment = ""
    current_segment_tokens = []

    for round_text in rounds:
        round_tokens = tokenizer(round_text)["input_ids"]

        # If a single round exceeds 2048 tokens, discard it
        if len(round_tokens) > MAX_TOKENS:
            print(f"🚨 A single round is too long ({len(round_tokens)} tokens), discarding this round.")
            continue  

        # Attempt to merge into the current segment
        combined_tokens = current_segment_tokens + round_tokens
        if len(combined_tokens) <= MAX_TOKENS:
            current_segment += round_text
            current_segment_tokens.extend(round_tokens)
        else:
            # Store the current segment and start a new one
            if current_segment:
                new_samples.append(current_segment.strip())
                print(f"✅ New sample generated! Token count: {len(current_segment_tokens)}")

            current_segment = round_text
            current_segment_tokens = round_tokens

    # Store the last segment
    if current_segment:
        new_samples.append(current_segment.strip())
        print(f"✅ Final sample generated! Token count: {len(current_segment_tokens)}")

    return {"formatted_text": new_samples}  # ✅ Ensure it returns a dict

print("Splitting long samples...")

# First, use `map()` to process the dataset
split_results = dataset["train"].map(split_conversation_by_rounds)

# Flatten `list[list[dict]] → list[dict]`
split_results_list = sum(split_results["formatted_text"], [])

# Convert back to Dataset
dataset = Dataset.from_list([{"formatted_text": text} for text in split_results_list])
# Ensure it remains a DatasetDict
dataset = DatasetDict({"train": dataset})  # ✅ Ensure "train" exists

print(f"✅ Processed dataset size: {len(dataset['train'])} (should be greater than 6000)")

# === 7️⃣ Split Training & Test Sets ===
train_data = dataset["train"].select(range(len(dataset["train"]) - 100))
test_data = dataset["train"].select(range(len(dataset["train"]) - 100, len(dataset["train"])))

print(f"✅ Train set: {len(train_data)} samples")
print(f"✅ Test set: {len(test_data)} samples")

# === 8️⃣ Save the Processed Dataset ===
new_dataset_dict = DatasetDict({"train": train_data, "test": test_data})

print(f"Saving processed dataset to {SAVE_PATH}...")
new_dataset_dict.save_to_disk(SAVE_PATH)
print(f"✅ Dataset saved at {SAVE_PATH}")

# === 9️⃣ Reload & Validate Dataset ===
print("Reloading dataset for validation...")
dataset = load_from_disk(SAVE_PATH)

print(f"✅ Train set size: {len(dataset['train'])}")
print(f"✅ Test set size: {len(dataset['test'])}")
print(f"✅ Last test sample:\n{dataset['test'][-1]['formatted_text']}")
