"""
Hyperparameter tuning script used on a sample of the validation dataset.
We must use a sample because training on 100,000+ rows takes too long.


"""
import torch
import optuna
import json
import random
from datasets import Dataset
from rapidfireai.automl import RFSFTConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model


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

  # Manual masking: -100 for prompt tokens, keep for response tokens
  labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
  # labels = labels[:len(full_ids)]  # Truncate to match input length

  # print(f"***{full_text}***\n")

  # debugging_dic = {}
  # for word in full_text.split():
  #   word_tokenized = tokenizer.encode(word, add_special_tokens=False)
  #   debugging_dic[word] = word_tokenized

  # print(f"***Full text: \n {full_text}***\n")
  # for key, value in debugging_dic.items():
  #   print(f"{key}: {value}")

  # print(f"***Labels: \n {labels}***\n")

  return {
    "input_ids": full_ids,
    "attention_mask": [1] * len(full_ids),
    "labels": labels
  }



validation_data = None
with open("validation.json", "r") as f:
  validation_data = json.load(f)

training_data_no_tags = None
with open("train_no_tags.json", "r") as f:
  training_data_no_tags = json.load(f)

training_data_all_else = None
with open("train_all_else.json", "r") as f:
  training_data_all_else = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
print(f"Pad token: {tokenizer.pad_token_id}")
print(f"EOS token: {tokenizer.eos_token}")

random.seed(42) #guarantees trial trains on identical 1000 rows

balanced_tuning_sample = random.sample(training_data_no_tags, 1250) + random.sample(training_data_all_else, 3750)
random.shuffle(balanced_tuning_sample)

validation_examples = [preprocess_function(example) for example in random.sample(validation_data, 500)] #200
training_examples = [preprocess_function(example) for example in balanced_tuning_sample] #1000

validation_dataset_sample = Dataset.from_list(validation_examples)
training_dataset_sample = Dataset.from_list(training_examples)

#Run lora
def create_model(trial=None):
  model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa" #flash_attention_2" # FASTER training on Colab GPUs
    # REMOVED: use_mamba_kernels=True (This causes the crash)
  )

  # Prepare for LoRA
  #model = prepare_model_for_kbit_training(model)
  #r_val = trial.suggest_categorical("r", [8, 16, 32, 64]) if trial else 16
  lora_dropout_val = trial.suggest_float("dropout", 0.01, 0.1) if trial else 0.05


  peft_config = LoraConfig(
      r=64,
      lora_alpha=128,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
      lora_dropout=lora_dropout_val,
      use_rslora=True,
      bias="none",
      task_type="CAUSAL_LM",
  )
  return get_peft_model(model, peft_config)


def objective(trial):
  return {
    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
  }

class CausalLMCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        input_ids = pad_sequence(input_ids, batch_first=True,
                                  padding_value=self.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True,
                                       padding_value=0)
        labels = pad_sequence(labels, batch_first=True,
                               padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)

# response_template = "<|assistant|>\n"
# response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

# collator = DataCollatorForCompletionOnlyLM(
#     response_template=response_template_ids,
#     tokenizer=tokenizer
# )

# collator = DataCollatorForLanguageModeling(
#   tokenizer=tokenizer,
#   mlm=False  # False for causal LM, True for BERT-style MLM
# )

trainer = Trainer(
    model=None,
    model_init=create_model,
    args = TrainingArguments(
      output_dir="./tuning_results",
      num_train_epochs=1,              
      per_device_train_batch_size=8,   
      gradient_accumulation_steps=4,   
      eval_strategy="steps",     
      eval_steps=100,
      save_strategy="no",
      logging_steps=10,
      gradient_checkpointing=True, #may delete later
      bf16=True,                      
      optim="adamw_8bit", 
    ),
    train_dataset=training_dataset_sample,
    eval_dataset=validation_dataset_sample,
    processing_class=tokenizer, #redundant but not harmful
    data_collator=collator
)

best_run = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=objective,
    n_trials=20,
    study_name="transformers_optuna_study2",
    storage="sqlite:///optuna_trials.db",
    pruner=optuna.pruners.MedianPruner(),
    load_if_exists=True
)

print(best_run)

