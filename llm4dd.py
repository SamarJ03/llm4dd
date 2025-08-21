#!/usr/bin/env python3 
import os, argparse, getpass, litellm
from litellm import completion, utils
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from utils import Log, CFG, Secrets
cfg = CFG(); log = Log(); sec = Secrets()

accepted_providers = ['openai', 'anthropic', 'xai', 'huggingface', 'openrouter', 'novita-ai']

class LLM: 
    def get_task_models() -> dict:
        tasks = cfg.get("models.tasks", {})
        valid_tasks = {}

        def validate_model(config: dict) -> tuple:
            provider = config.get('provider')
            model = config.get('model')
            api_key = sec.get(f"{provider.upper()}_API_KEY")
            return (litellm.utils.check_valid_key(model=model, api_key=api_key), {'provider': provider, 'model': model, 'params': config.get('params', {})})

        for name, config in tasks.items():
            is_valid, model_config = validate_model(config)
            if not is_valid: is_valid, model_config = validate_model(config['fallback'])

            if is_valid:
                valid_tasks[name] = model_config
                logger.debug(f"Model validated for task: {name}")
            else: logger.error(f"No valid model found for task: {name}")
        return valid_tasks
    
def CLI() -> argparse.ArgumentParser: 
    parser = argparse.ArgumentParser(prog='l4d', description='Command line interface for LLM4DD')

    parser.add_argument('--verbose', '-v',
        type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        default="INFO", help="Choose verbosity level."
    )
    parser.add_argument('--source-path', '-sp',
        type=str, default="LOCAL",
        help="Configure path to source data. Defaults to the local path at 'data/'."
    )
    parser.add_argument('--api-keys', '-k',
        type=str, default='load', choices=['set', 'load'], 
        help="Configure API keys. [load] pulls keys from '~/.config/llm4dd/keys.json'. [set] allows you to configure them through the command line."
    )

    args = parser.parse_known_args(('verbose', 'api_keys', 'source_path'))
    if args.verbose:
        log.changeVerbose(args.verbose)
        logger.info(f"Verbosity level set to {args.verbose}")

    if str(args.source_path).lower() != "local":
        if not Path(args.source_path).exists(): raise FileExistsError(f"file not found:{args.source_path}")
        cfg.set('env.source_path', args.source_path)
        logger.info(f"Source path set to: {args.source_path}")
    else:
        default_path = Path(__file__) / "data"
        if not default_path.exists():
            default_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created default data directory: {default_path}")
        cfg.set('env.source_path', str(default_path))
        logger.info(f"Using default source path: {default_path}")

    if args.api_keys == 'load':
        keys = sec.load()
        if keys:
            logger.info("API keys loaded successfully")
            for key in sec.list(): logger.debug(f"keys found: {sec.list()}")
        else: logger.warning("No API keys found in configuration")
    elif args.api_keys == 'set':
        print("Enter 'q' at any time to finish entering API keys..")
        keys_set = 0
        while True: 
            provider = input("Enter LLM API provider: ").lower().strip()
            if provider=='q': break
            elif provider not in accepted_providers: print(f'Invalid provider. Please enter API keys from the supported providers: {accepted_providers}')
            key = getpass.getpass(prompt="Enter API key: ").strip()
            if key.lower()=='q': break
            sec.set(name=f'{provider.upper()}_API_KEY', value=key); keys_set+=1
        if keys_set==0: raise Exception(f'No keys set..')
        else: logger.info(f'Succesfully set LLM API keys..')
    
    return parser

if __name__ == "__main__": 
    parser: argparse.ArgumentParser = CLI()
    # subparsers = parser.add_subparsers(title='cmd')
    # modelParser = subparsers.add_parser(name="llm")

    task_models = LLM.get_task_models()
