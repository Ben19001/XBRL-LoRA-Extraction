This was an experiment conducted for the purposes of examining the performance of LoRA fine-tuning on a decoder-only model such as
Qwen/Qwen3-4B-Instruct-2507 for XBRL finanical entity recognition and classification.  

Prerequisites:
- A hugging face account with an access token
- Access to a Google Colab environment (scripts are configured for this setup) or an A100 GPU

Instructions: 
1. Run both scripts in the datasets folder.
2. Run fifty_fifty.py in the hyperparameter_searches folder because this performed better than the 75% positive tags/25% negative tags in three_to_one.py.
3. Run fifty_fifty.py in the training folder. You will have to substitute the learning rate and dropout the results of your hyperparameter search yielded.
4. Run the evaluation script inside the evaluation folder. You need to point vllm to the location of your adapter files on HuggingFace. 
