"""
Final training script
"""

import torch
import json
import random
from datasets import Dataset
from rapidfireai.automl import RFSFTConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from google.colab import drive, userdata
from huggingface_hub import login

drive.mount('/content/drive')
hf_token = userdata.get('HF_TOKEN')
login(hf_token)


def preprocess_function(example):
  prompt_only = (
      f"<|system|>\n"
      f"{example['instruction']}\n"
      f"<|user|>\n"
      f"{example['input']}\n"
      f"<|assistant|>\n"
  )
  full_text = prompt_only + example['output'] + tokenizer.eos_token
  prompt_ids = tokenizer.encode(prompt_only, add_special_tokens=False)
  full_ids = tokenizer.encode(full_text, add_special_tokens=False)
  labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
  return {
      "input_ids": full_ids,
      "attention_mask": [1] * len(full_ids),
      "labels": labels
  }

testing_data = None
with open("test.json", "r") as f:
    testing_data = json.load(f)

training_data_no_tags = None
with open("train_no_tags.json", "r") as f:
    training_data_no_tags = json.load(f)

training_data_all_else = None
with open("train_all_else.json", "r") as f:
    training_data_all_else = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

random.seed(42)

balanced_tuning_sample = random.sample(training_data_no_tags, 50000) + random.sample(training_data_all_else, 150000)
random.shuffle(balanced_tuning_sample)

testing_examples = [preprocess_function(example) for example in random.sample(testing_data, 2000)]
training_examples = [preprocess_function(example) for example in balanced_tuning_sample]

testing_dataset_sample = Dataset.from_list(testing_examples)
training_dataset_sample = Dataset.from_list(training_examples)

def create_model():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.09448383910393976, #insert from hyperparameter tuning script
        use_rslora=True,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)

class CausalLMCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)

final_model = create_model()
trainer = Trainer(
    model=final_model,
    args=TrainingArguments(
        output_dir="/content/drive/MyDrive/tuning_results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=0.0002265716563026846, #insert from hyperparameter tuning script
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        logging_steps=10,
        gradient_checkpointing=True,
        bf16=True,
        optim="adamw_8bit",
        push_to_hub=True,
        hub_model_id="<INSERT_HF_MODEL_ID>",
        hub_private_repo=True,
        load_best_model_at_end=True
    ),
    train_dataset=training_dataset_sample,
    eval_dataset=testing_dataset_sample,
    processing_class=tokenizer,
    data_collator=collator
)

print("Starting training...")
trainer.train()