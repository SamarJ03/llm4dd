import pandas as pd
import argparse
import random
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import os
import json
from tqdm import tqdm

base_dir = '/Users/samarjosyula/Desktop/PROJECTS/pgccInhibitorDrugDiscovery'
resource_path = os.path.join(base_dir, 'resources')
os.makedirs(resource_path, exist_ok=True)
scaffoldDataset_path = os.path.join(f'{base_dir}/data', 'scaffold_datasets')
os.makedirs(scaffoldDataset_path, exist_ok=True)
promptFile_path = os.path.join(resource_path, 'prompt_file')
os.makedirs(promptFile_path, exist_ok=True)

class Inferencer: 
    def run(self):
        args = self.args
        if args['dataset'] in [
                'E-State', 'Functional Group', 'Molecular Topology', 
                'Fingerprint Based', 'Surface Area', 
                'Structural', 'Physiochemical'
            ]:
            file_folder = os.path.join(args['input_folder'], 'rdkit')
        else:
            file_folder = os.path.join(args['input_folder'], args['dataset'])
        if args['subtask'] == "":
            train_file_name = args['dataset'] + '_train.csv'
        else:
            train_file_name = args['subtask'] + '_train.csv'
        train_file_path = os.path.join(file_folder, train_file_name)
        smile_label_list = self.load_dataset(train_file_path)
        tokenizer, pipeline = self.get_hf_tokenizer_pipeline(args['model'])
        dk_prompt = self.get_inference_prompt()

        list_of_smile_label_lists = self.split_smile_list(smile_label_list, dk_prompt, tokenizer, args['list_num'])
        print(f'Split into {len(list_of_smile_label_lists)} lists')

        output_file_folder = os.path.join(resource_path, args['output_folder'], args['model'], args['dataset'])
        if args['subtask']:
            output_file_name = f"{args['model']}_{args['dataset']}_{args['subtask']}_dk_response_sample_{args['list_num']}.txt"
        else:
            if args['dataset'] in [
                'E-State', 'Functional Group', 'Molecular Topology', 'Fingerprint Based', 
                'Surface Area', 'Structural', 'Physiochemical'
            ]:
                output_file_name = f"{args['model']}_rdkit_{args['dataset']}_dk_response_sample_{args['list_num']}.txt"
                output_file_folder = os.path.join(resource_path, args['output_folder'], args['model'], 'rdkit')
                # ex: ~/LLM4SD_resources/inference_model_response/chemdfm/rdkit/chemdfm_rdkit_E-State_dk_response_sample_30.txt
            else:
                output_file_name = f"{args['model']}_{args['dataset']}_dk_response_sample_{args['list_num']}.txt"
                # ex: ~/LLM4SD_resources/inference_model_response/chemdfm/maccs/chemdfm_maccs_dk_response_sample_30.txt

        os.makedirs(output_file_folder, exist_ok=True)
        output_file = os.path.join(output_file_folder, output_file_name)

        print(f'Start getting response from model {args["model"]}....')
        response_list = self.get_model_response(args['model'], list_of_smile_label_lists, pipeline, dk_prompt, tokenizer)
        with open(output_file, 'w') as f:
            for response in response_list:
                f.write(response)
                f.write("\n\n================================\n\n")

    # def __init__(self, args=None, dataset=None, subtask=None, model=None, prompt_folder=None, prompt_file=None,
    #              input_folder=None, output_folder=None, list_num=None, verbose=False):
    #     if isinstance(args, dict): # args parameter is already in Dict format
    #         newArgs = args.copy()
    #         newArgs.setdefault('dataset', 'rdkit')
    #         newArgs.setdefault('subtask', '')
    #         newArgs.setdefault('model', 'galactica-6.7b')
    #         newArgs.setdefault('prompt_folder', promptFile_path)
    #         newArgs.setdefault('prompt_file', 'inference_prompt.json')
    #         newArgs.setdefault('input_folder', scaffoldDataset_path)
    #         newArgs.setdefault('output_folder', 'inference_model_response')
    #         newArgs.setdefault('list_num', 30)
    #         newArgs.setdefault('verbose', False)
    #         args = newArgs
    #     else: # args parameter is argparse.Namespace or None
    #         parser = argparse.ArgumentParser()
    #         parser.add_argument('--dataset', type=str, default='rdkit')
    #         parser.add_argument('--subtask', type=str, default='')
    #         parser.add_argument('--model', type=str, default='galactica-6.7b')
    #         parser.add_argument('--prompt_folder', type=str, default='inference_prompt')
    #         parser.add_argument('--prompt_file', type=str, default='inference_prompt.json')
    #         parser.add_argument('--input_folder', type=str, default=scaffoldDataset_path)
    #         parser.add_argument('--output_folder', type=str, default='inference_model_response')
    #         parser.add_argument('--list_num', type=int, default=30)
    #         parser.add_argument('--verbose', type=bool, default=False)
    #         parsedArgs = parser.parse_args()
    #         args = {
    #             'dataset' : parsedArgs.dataset,
    #             'subtask' : parsedArgs.subtask,
    #             'model': parsedArgs.model,
    #             'prompt_folder': parsedArgs.prompt_folder,
    #             'prompt_file': parsedArgs.prompt_file,
    #             'input_folder': parsedArgs.input_folder,
    #             'output_folder' : parsedArgs.output_folder,
    #             'list_num': parsedArgs.list_num,
    #             'verbose': parsedArgs.verbose
    #         }

    #     # optional overrides and error handling
    #     dataset = dataset.lower()
    #     if dataset and dataset in ('ecfp4', 'maccs', 'rdkit'): args['dataset'] = dataset
    #     else: raise NotImplementedError(f'{dataset} is not a valid descriptor type..')
    #     if subtask: args['subtask'] = subtask
    #     if model: args['model'] = model
    #     if prompt_folder: args['prompt_folder'] = prompt_folder
    #     if prompt_file: args['prompt_file'] = prompt_file
    #     if input_folder: args['input_folder'] = input_folder
    #     if output_folder: args['output_folder'] = output_folder
    #     if list_num: args['list_num'] = list_num

    #     self.dataset = args['dataset']
    #     self.subtask = args['subtask']
    #     self.model = args['model']
    #     self.prompt_folder = args['prompt_folder']
    #     self.prompt_file = args['prompt_file']
    #     self.input_folder = args['input_folder']
    #     self.output_folder = args['output_folder']
    #     self.list_num = args['list_num']
    #     self.args = args

    def __init__():
        pass

    