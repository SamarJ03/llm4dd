#!/usr/bin/env python3 
import os, argparse, getpass, litellm
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
from litellm import completion, utils
from utils import Log, CFG, Secrets, LLM
cfg = CFG(); log = Log(); sec = Secrets(); llm = LLM()

# Supported LLM providers
accepted_providers = ['openai', 'anthropic', 'xai', 'huggingface', 'openrouter', 'novita-ai']

# class Model:
#     def __init__(self, provider: str, model: str, params: Optional[Dict] = None) -> None:
#         self.provider = provider; self.model = model
#         self.params = params or {}
#         self.api_key = sec.get(f"{provider.upper()}_API_KEY")        
#         if not self.api_key: raise ValueError(f"No API key found for provider: {provider}")
        
#     def call(self, messages: List[Dict[str, str]]) -> Dict:
#         try:
#             return completion(
#                 model=self.model,
#                 messages=messages,
#                 api_key=self.api_key,
#                 **self.params
#             )
#         except Exception as e:
#             logger.error(f"API call failed: {str(e)}")
#             raise

class LLM:
    def __init__(self, models_cfg:dict): 
        self.models_cfg = models_cfg
        
    def _parse_models(self): pass
    
    class Model: 
        def __init__(self, model: str, provider: str): 
            self.model = model; self.provider = provider
            self.key:str = sec.get(name=provider)
            if not self.validate: raise Exception(f'Invalid API key: {self.provider}')
            
        def call(self, context:list[str]): 
            litellm.api_key = self.key
            response = litellm.completion(messages=context, model=self.model)
            #TODO left off here @ 4:53PM on 8/25

        def validate(self): return litellm.utils.check_valid_key(model=self.model, api_key=self.key)

def CLI() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='l4d',
        description='Command line interface for LLM4DD'
    )

    parser.add_argument('--verbose', '-v',
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Choose verbosity level."
    )
    parser.add_argument('--source-path', '-sp',
        type=str,
        default="LOCAL",
        help="Configure path to source data. Defaults to the local path at 'data/'."
    )
    parser.add_argument('--api-keys', '-k',
        type=str,
        default='load',
        choices=['set', 'load'],
        help="Configure API keys. [load] pulls keys from '~/.config/llm4dd/keys.json'. [set] allows you to configure them through the command line."
    )

    args, _ = parser.parse_known_args()
    if args.verbose:
        log.changeVerbose(args.verbose)
        logger.info(f"Verbosity level set to {args.verbose}")

    if str(args.source_path).lower() != "local":
        if not Path(args.source_path).exists():
            raise FileNotFoundError(f"Source path not found: {args.source_path}")
        cfg.set('env.source_path', args.source_path)
        logger.info(f"Source path set to: {args.source_path}")
    else:
        default_path = Path(__file__).parent / "data"
        default_path.mkdir(parents=True, exist_ok=True)
        cfg.set('env.source_path', str(default_path))
        logger.info(f"Using default source path: {default_path}")

    def _configure_api_keys() -> None:
        print("Enter 'q' at any time to finish entering API keys..")
        keys_set = 0
        while True:
            provider = input("Enter LLM API provider: ").lower().strip()
            if provider == 'q':
                break
            if provider not in accepted_providers:
                print(f'Invalid provider. Please enter API keys from the supported providers: {accepted_providers}')
                continue

            key = getpass.getpass(prompt="Enter API key: ").strip()
            if key.lower() == 'q':
                break

            sec.set(name=f'{provider.upper()}_API_KEY', value=key)
            keys_set += 1

        if keys_set == 0: raise ValueError("No API keys were set")
        logger.info(f'Successfully set {keys_set} LLM API key(s)')

    if args.api_keys == 'load':
        keys = sec.load()
        if keys:
            logger.info("API keys loaded successfully")
            for key in sec.list():
                logger.debug(f"Key found: {key}")
        else:
            logger.warning("No API keys found in configuration")
    elif args.api_keys == 'set':
        _configure_api_keys()

    return parser

if __name__ == "__main__":
    parser = CLI()
    task_models: dict = CFG.get_task_models()

    # Add subcommands here when needed
    # subparsers = parser.add_subparsers(title='cmd')
    # model_parser = subparsers.add_parser(name="llm")
    