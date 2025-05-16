from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel
import torch
import pandas as pd
import json
import argparse
import random



def load_model(model_name: str, use_vanilla: bool = False):

    if model_name == "./TG-LLM/TGQA_TGR/final":
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the model
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map="auto", torch_dtype=torch.float16)
        # Resize the model and reload the model with the weight
        model.resize_token_embeddings(len(tokenizer))

        if not use_vanilla:
            model = PeftModel.from_pretrained(model, model_name)

    elif model_name == "./temp_t5_l2_cbqa":

        if use_vanilla:
            tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
            model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base") 
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    return tokenizer, model


parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=['current', 'basic'], type=str)
parser.add_argument("--model", type=str, choices=['tg_llm', 'tempt5'])
parser.add_argument("--name", type=str, default="tg_llm")
parser.add_argument("--vanilla", action="store_true", help="Use vanilla model")

args = parser.parse_args()
random.seed(42)
output_f = f"{args.name}.jsonl"
cleaned_f = output_f.replace(".jsonl", "_cleaned.jsonl")

if args.model == 'tg_llm':  
    model_name = "./TG-LLM/TGQA_TGR/final"  # Replace with your desired model name
elif args.model == 'tempt5':
    model_name = "./temp_t5_l2_cbqa"

tokenizer, model = load_model(model_name, use_vanilla=args.vanilla)


# prepare the data
if args.task == 'current':  
    file_path = f"/home/soyeon/temp_eval/results/dyknow/ours_vs_dyknow/gemma_temp_align.jsonl"
    with open(file_path, 'r') as file:
        qa_data = [json.loads(line) for line in file]

    for q_items in qa_data:

        prompt = q_items['question']

        print(prompt)
        # Tokenize and move to device
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()} 


        if args.model == 'tempt5' and args.vanilla:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            outputs = model.generate(input_ids)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        else:
            # Generate output
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Decode output (only the newly generated tokens)


            if args.model == 'tempt5':
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif args.model == 'tg_llm':
                generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)


        print("Answer:", generated_text)


        # save responses to log file (current)
        qa_result_dict = {'idx': q_items['idx'], \
                        'question': prompt, \
                        'model_response': generated_text, \
                        'gold_answer': q_items['gold_answer'], \
                        'outdated': q_items['outdated']} 
        

        # save as jsonl file
        with open(output_f, 'a') as file:
            file.write(json.dumps(qa_result_dict) + '\n')



else:
    file_path = f"/home/soyeon/temp_eval/results/dyknow/basic_new/gemma_basic.jsonl"
    with open(file_path, 'r') as file:
        qa_data = [json.loads(line) for line in file]

    none = [x for x in qa_data if x['qtype'] == 'basic_none']
    unique = [y for y in qa_data if y['qtype'] == 'basic_unique']
    multiple = [z for z in qa_data if z['qtype'] == 'basic_multiple']


    for curr_cat in [none, unique, multiple]:
        # sample only 100 questions

        curr_cat = random.sample(curr_cat, 100)
        for q_items in curr_cat:

            # sample only 100 questions
            

            prompt = q_items['question']

            print(prompt)
            # Tokenize and move to device
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()} 

            # Generate output
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Decode output (only the newly generated tokens)
            if args.model == 'tempt5':
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif args.model == 'tg_llm':
                generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            print("Answer:", generated_text)


            # save responses to log file
            qa_result_dict = {'idx': q_items['idx'], \
                            "qtype": q_items['qtype'], \
                            "op": q_items['op'], \
                            "sql_for_db": q_items['sql_for_db'], \
                            'question': prompt, \
                            'model_response': generated_text, \
                            'correct_rows': q_items['correct_rows'],\
                            'incorrect_rows': q_items['incorrect_rows']}

            # save as jsonl file
            with open(output_f, 'a') as file:
                file.write(json.dumps(qa_result_dict) + '\n')
            






# clean files
seen = set()
unique_lines = []


with open(output_f, "r") as f:
    for line in f:
        if line.strip() == "":
            continue
        obj = json.loads(line)
        # Serialize object to a string (excluding fields like idx if needed)
        obj_str = json.dumps(obj, sort_keys=True)
        if obj_str not in seen:
            seen.add(obj_str)
            unique_lines.append(obj)

# Write unique lines back
with open(cleaned_f, "w") as f:
    for obj in unique_lines:
        f.write(json.dumps(obj) + "\n")
