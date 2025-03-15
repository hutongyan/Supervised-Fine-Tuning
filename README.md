# Supervised-Fine-Tuning

## Overreview
This project focuses on fine-tuning the Pythia-160M model for conversational AI using domain-specific dialogue datasets deita-6k-v0. 

## Implement Details

### Dataset loading 
The dataset is downloaded and stored using datasets.

```
from datasets import load_dataset
dataset = load_dataset("hkust-nlp/deita-6k-v0")
dataset.save_to_disk("./deita-6k-v0")
```
### Dataset preprocessing
In the preprocessing stage, we need to format conversational data using Mistral-7B-Instruct-v0.1 chat template and pack multi-turn dialogues to meet the training context length (2048)
The key column formatted_text is extracted and tokenized using Hugging Face's AutoTokenizer.
Padding and truncation ensure input sequences stay within 2048 tokens.
Split the preprocessed dataset into two parts, leaving the last 100 instances as test dataset.

### Model Fine-Tuning
The original pre-trained model is downloaded on huggingface and loaded from a local path.

Training is conducted on multi-GPU (CUDA_VISIBLE_DEVICES) using DeepSpeed for memory optimization.

Uses DeepSpeed for multi-GPU training.

Runs for 3 epochs with batch size 32 per GPU.

The learning rate is 2e-5, optimized for Pythia-160M.

Evaluation and checkpointing occur at the end of each epoch.

The Trainer API from transformers is used for training.

The DataCollatorForLanguageModeling ensures proper tokenization and padding.

Fine-tuning begins with trainer.train(), logging loss and evaluation metrics.

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py
```

## Inference and Evaluation
After training, BLEU scores are computed to evaluate model performance.
The model generates predictions for the test set.
Predictions are compared to reference responses using BLEU scoring.
In my experiment: the Average BLEU Score is 0.8371
