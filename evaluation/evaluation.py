
"""
Final evaluation script
"""

import os
import random
import json
import re
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from google.colab import userdata
from huggingface_hub import login
from collections import Counter


login(token=userdata.get('HF_TOKEN'))

llm = LLM(
  model="Qwen/Qwen3-4B-Instruct-2507",
  trust_remote_code=True,
  enable_lora=True,
  max_lora_rank=16,           
  dtype="bfloat16",
  max_model_len=8192,
  gpu_memory_utilization=0.9, 
)

lora_request = LoRARequest(
    lora_name="xbrl_adapters", #may change later
    lora_int_id=1,
    lora_path="Ben16001/XBRL-LoRA300k-PlanB",
)

global_true_positives = 0
global_false_positives = 0
global_false_negatives = 0
macro_f1_dic = {}

def determine_counts(generated_text, actual_text):
  actual_tags = Counter([line.strip() for line in actual_text.split("\n") if line.strip()])
  generated_tags = Counter([line.strip() for line in generated_text.split("\n") if line.strip()])

  if len(actual_tags) == 1 and "No XBRL tags found" in actual_tags:
    if len(generated_tags) == 1 and "No XBRL tags found" in generated_tags:
      return 0, 0, 0 #majority of cases

  true_positives = 0
  false_positives = 0
  false_negatives = 0

  if actual_tags == generated_tags:
    for generated_tag_key, generated_tag_value in generated_tags.items():
      label = generated_tag_key.split(":", 1)[0].strip()
      lst = macro_f1_dic.setdefault(label, [0, 0, 0])
      lst[0] += int(generated_tag_value)
      true_positives += int(generated_tag_value)
    return true_positives, 0, 0

  if "No XBRL tags found" in actual_tags.keys() and "No XBRL tags found" not in generated_tags.keys():
    for generated_tag_key, generated_tag_value in generated_tags.items():
      if generated_tag_key != "No XBRL tags found":
        label = generated_tag_key.split(":", 1)[0].strip()
        lst = macro_f1_dic.setdefault(label, [0, 0, 0])
        lst[1] += int(generated_tag_value)
        false_positives += int(generated_tag_value)
    return true_positives, false_positives, false_negatives
  elif "No XBRL tags found" not in actual_tags.keys() and "No XBRL tags found" in generated_tags.keys():
    for actual_tag_key, actual_tag_value in actual_tags.items():
      if actual_tag_key != "No XBRL tags found":
        label = actual_tag_key.split(":", 1)[0].strip()
        lst = macro_f1_dic.setdefault(label, [0, 0, 0])
        lst[2] += int(actual_tag_value)
        false_negatives += int(actual_tag_value)
    return true_positives, false_positives, false_negatives

  intersection = actual_tags & generated_tags
  for intersection_tag_key, intersection_tag_value in intersection.items():
    label = intersection_tag_key.split(":", 1)[0].strip()
    lst = macro_f1_dic.setdefault(label, [0, 0, 0])
    lst[0] += int(intersection_tag_value)
    true_positives += int(intersection_tag_value)

  false_negatives_dic = actual_tags - generated_tags
  false_positives_dic = generated_tags - actual_tags

  print(false_negatives_dic)
  print(false_positives_dic)

  for false_positive_key, false_positive_value in false_positives_dic.items():
    label = false_positive_key.split(":", 1)[0].strip()
    lst = macro_f1_dic.setdefault(label, [0, 0, 0])
    lst[1] += int(false_positive_value)
    false_positives += int(false_positive_value)

  for false_negative_key, false_negative_value in false_negatives_dic.items():
    label = false_negative_key.split(":", 1)[0].strip()
    lst = macro_f1_dic.setdefault(label, [0, 0, 0])
    lst[2] += int(false_negative_value)
    false_negatives += int(false_negative_value)

  return true_positives, false_positives, false_negatives


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")


def preprocess_function(examples):
  all_prompts = []
  for example in examples:
    prompt_only = (
      f"<|system|>\n"
      f"{example['instruction']}\n"
      f"<|user|>\n"
      f"{example['input']}\n"
      f"<|assistant|>\n"
    )
    all_prompts.append(tokenizer.encode(prompt_only, add_special_tokens=False))

  outputs = llm.generate(
    [{"prompt_token_ids": p} for p in all_prompts],
    lora_request = lora_request,
    sampling_params = SamplingParams(
        temperature = 0.1,
        max_tokens = 2048,
    ),
  )
  return outputs



validation_data = None
with open("validation.json", "r") as f:
    validation_data = json.load(f)

random.seed(42)

examples = random.sample(validation_data, 2000)
llm_outputs = preprocess_function(examples)
for example, llm_output in zip(examples, llm_outputs):
  generated_text = llm_output.outputs[0].text
  true_positives, false_positives, false_negatives = determine_counts(generated_text, example['output'])
  global_true_positives += true_positives
  global_false_positives += false_positives
  global_false_negatives += false_negatives


micro_precision = global_true_positives / (global_true_positives + global_false_positives) if (global_true_positives + global_false_positives) > 0 else 0
micro_recall = global_true_positives / (global_true_positives + global_false_negatives) if (global_true_positives + global_false_negatives) > 0 else 0

micro_f1 = 2 * (micro_precision * micro_recall / (micro_precision + micro_recall))

macro_precision_averages = []
macro_recall_averages = []
for key, value in macro_f1_dic.items():
  macro_precision_average = 0
  macro_recall_average = 0
  if value[0] + value[1] != 0:
    macro_precision_average = value[0] / (value[0] + value[1])
  if value[0] + value[2] != 0:
    macro_recall_average = value[0] / (value[0] + value[2])
  macro_precision_averages.append(macro_precision_average)
  macro_recall_averages.append(macro_recall_average)

final_macro_precision_average = sum(macro_precision_averages) / len(macro_precision_averages)
final_macro_recall_average = sum(macro_recall_averages) / len(macro_recall_averages)

macro_f1 = 2 * (final_macro_precision_average * final_macro_recall_average / (final_macro_precision_average + final_macro_recall_average))

print(f"Fine tuned model achieved a macro-F1 of: {macro_f1}")
print(f"Fine tuned model achieved a micro-F1 of: {micro_f1}")