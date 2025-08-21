import sys 
import os
import json
import time
import argparse
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM, GenerationConfig
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
# from transformers import tokeniz  

# prompt_path = os.path.join('resources', 'prompts')
# dataset_path = os.path.join('resources', 'scaffold_features')
# if not os.path.exists(prompt_path): raise Exception('Prompt dir not found..')
# if not os.path.exists(dataset_path): raise Exception('Dataset dir not found..')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Synthesizer:
    def getTokenizer(self, model, is8bit=False): 
        hf_model = None
        if model == 'falcon-7b': hf_model = "tiiuae/falcon-7b-instruct"
        elif model == 'falcon-40b': hf_model = "tiiuae/falcon-40b-instruct"
        elif model == "galactica-6.7b": hf_model = "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k"
        elif model == "galactica-30b": hf_model = "GeorgiaTechResearchInstitute/galactica-30b-evol-instruct-70k"
        elif model == "chemllm-7b": hf_model = "AI4Chem/ChemLLM-7B-Chat"
        elif model == "chemdfm": hf_model = "X-LANCE/ChemDFM-13B-v1.0"
        else: raise NotImplementedError(f"Cannot find Hugging Face tokenizer for model {model}.")
        if model in ('falcon-40b', 'galactica-30b'): is8bit = True
        kwargs = {}
        quantization_config = None
        if is8bit: quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=200.0,)
        if model=='chemllm-7b':
            pipeline = AutoModelForCausalLM.from_pretrained(
                hf_model, torch_dtype=torch.float16, 
                device_map='auto', trust_remote_code='True', 
            )
            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        elif model=='chemdfm':
            tokenizer = LlamaTokenizer.from_pretrained(hf_model)
            pipeline = LlamaForCausalLM.from_pretrained(hf_model, torch_dtype=torch.float16, device_map='auto')
        else:
            kwargs['quantization_config'] = quantization_config
            tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=False)
            pipeline = transformers.pipeline(
                'text-generation',
                model=hf_model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                use_fast=False,
                device_map='auto',
                model_kwargs=kwargs
            )
        return tokenizer, pipeline

    def getPrompt(self):
        prompt_file = os.path.join(self.prompt_path)
        if not os.path.exists(prompt_file): raise Exception(f'Could not verify prompt file path:\n{self.prompt_path}')
        prompts, tasks = [], []
        with open (prompt_file) as f: prompt_dict = json.load(f)
        if self.model.lower() in ('falcon-7b', 'galactica-6.7b', 'chemllm-7b', 'chemdfm'): dataset_key = self.dataset + '_small'
        else: dataset_key = self.dataset + '_big'
        if self.verbose: print(f'Extracting {dataset_key} dataset prior knowledge prompt ....')
        if not self.subtask: 
            tasks.append(self.dataset)
            prompts.append(prompt_dict[dataset_key])
        elif self.subtask:  # for tox21 and sider
            if self.verbose: print(f"Extracting {self.subtask} task prior knowledge prompt ....")
            tasks.append(self.dataset + "_" + self.subtask )
            prompts.append(prompt_dict[dataset_key][self.subtask])
        else: raise NotImplementedError(f"""No prior knowledge prompt for task {self.dataset}.""")
        return tasks, prompts 

    def getModelResponse(self, model, tokenizer, pipeline, prompts):
        if model in ["galactica-6.7b", "galactica-30b", "chemllm-7b"]:
            system_prompt = ("Below is an instruction that describes a task. "
                            "Write a response that appropriately completes the request.\n\n"
                            "### Instruction:\n{instruction}\n\n### Response:\n")
        elif model in ['falcon-7b', 'falcon-40b']:
            system_prompt = "{instruction}\n"
        elif model in ["chemdfm"]:
            system_prompt = "Human: {instruction}\nAssistant:"
        else:
            system_prompt = "{instruction}\n"
        responses = []

        for prompt in prompts:
            input = system_prompt.format_map({'instruction': prompt.strip()})
            lenInput = len(tokenizer.tokenize(input))
            if self.verbose: print(input)
            max_new_tokens = self.getTokenLimit(model, for_response=True) - lenInput
            if model=='chemllm-7b':
                inputs = tokenizer(input, return_tensors='pt').to(device)
                generation_config = GenerationConfig(
                    do_sample=True,
                    top_k=1,
                    temperature=float(0.5),
                    max_new_tokens=max_new_tokens,
                    repitition_penalty=float(1.2),
                    pad_token_id=tokenizer.eos_token_id            
                )
                outputs = pipeline.generate(**inputs, generation_config=generation_config)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif model=='chemdfm':
                inputs = tokenizer(input, return_tensors='pt').to(device)
                generation_config = GenerationConfig(
                    do_sample=True,
                    top_k=20,
                    top_p=0.9,
                    temperature=0.9,
                    max_new_tokens=max_new_tokens,
                    repitition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id
                )
                outputs = pipeline.generate(**inputs, generation_config=generation_config)
                generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(input):]
            else: 
                text_generator = pipeline(
                    input, 
                    min_new_tokens=0,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=3, 
                    temperature=float(0.5),
                    repitition_penalty=float(1.2),
                    renormalize_logits=True
                )
                generated_text = text_generator[0]['generated_text']
            if model in ('galactica-6.7b', 'galactica-30b', 'chemllm-7b'): generated_text = generated_text.split('### Response:\n')[1]
            # elif model in ('falcon-7b', 'falcon-40b', 'chemdfm'): pass
            if self.verbose: print(generated_text)
            responses.append(generated_text)
        return responses

    @staticmethod
    def getTokenLimit(model, for_response=False):
        sel = ('falcon-7b', 'falcon-40b', 'galactica-6.7b', 'galactica-30b', 'chemdfm')
        if for_response: # For get response
            if model in sel: return 2048
            else: return 4096
        else: # For split input list
            if model in sel: return round(2048*3/4)
            else: return round(2048*3/4)

    def run(self):
        start = time.time()
        tokenizer, pipeline = self.getTokenizer(self.model)
        tasks, prompts = self.getPrompt()
        response_list = self.getModelResponse(self.model, tokenizer, pipeline, prompts)
        if self.subtask or self.subtask != '': subtask_name = "_" + self.subtask
        else: subtask_name = ''
        output_file = os.path.join(self.output_dir, f'{self.model}{subtask_name}_pk_response.txt')
        os.makedirs(output_file, exist_ok=True)
        with open(output_file, 'w') as f:
            for i in range(len(tasks)):
                f.write(f'task name: {tasks[i]}\n')
                f.write('Response from model: \n')
                f.write(response_list[i])
                f.write("\n\n================================\n\n")
        end = time.time()
        print(f"Synthesize/Time elapsed: {end-start} seconds")

    def __init__(self, args: argparse.Namespace):
        self.output_dir = args.output_dir
        self.verbose = args.verbose
        self.dataset = args.datset
        self.subtask = args.subtask
        self.model = args.model
        self.prompt_path = args.prompt_path

if __name__ == "__main__": Synthesizer.run()