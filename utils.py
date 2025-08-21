import os, sys, inspect, resource, argparse, json, yaml, shutil, getpass 
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any
from datetime import datetime

__all__ = ['Log', 'CFG', 'Secrets']

class Log:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.handlers: Dict[str, int] = {}
            logger.remove()
    
    def setLog(self, name: Optional[str] = None, verbose: str = os.getenv("VERBOSE", "INFO"), 
              debug: Optional[bool] = None, trackMem: Optional[bool] = None) -> Any:
        try:
            debug = debug if debug is not None else os.getenv("DEBUG", 'false').lower() == 'true'
            trackMem = trackMem if trackMem is not None else os.getenv("TRACK_MEM", 'false').lower() == 'true'

            if name is None:
                try:
                    frame = inspect.stack()[-1]
                    name = Path(frame.filename).name
                except Exception:
                    name = '<unknown>'

            base_path = Path(os.getenv("BASE_PATH", Path(__file__).parent.parent))
            log_path = base_path / "logs"
            debug_path = log_path / "debug"
            error_path = log_path / "error"

            for path in (log_path, debug_path, error_path):
                path.mkdir(parents=True, exist_ok=True)

            for handler_id in self.handlers.values():
                logger.remove(handler_id)
            self.handlers.clear()

            self.handlers["stderr"] = logger.add(
                sys.stderr,
                format="<level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                colorize=True,
                level=verbose.upper()
            )

            self.handlers["debug"] = logger.add(
                str(debug_path / f"{name}_debug.log"),
                rotation="500 MB",
                retention="10 days",
                compression="zip",
                level="DEBUG"
            )

            self.handlers["error"] = logger.add(
                str(error_path / f"{name}_error.log"),
                rotation="100 MB",
                retention="1 month",
                level="ERROR",
                backtrace=True,
                diagnose=True
            )

            if trackMem:
                self.handlers["memory"] = logger.add(
                    lambda msg: logger.opt(depth=1).debug(
                        f"Memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} MB"
                    ),
                    level="TRACE"
                )

            return logger

        except Exception as e:
            if not logger.handlers:
                logger.add(sys.stderr, level="ERROR")
            logger.error(f"Failed to setup logging: {e}")
            return logger

    def changeVerbose(self, newLevel: str) -> Any:
        """Change verbosity level for stderr handler."""
        if newLevel.upper() not in logger._levels:
            logger.warning(f"Invalid verbosity level: {newLevel.upper()}")
            return logger

        if "stderr" in self.handlers:
            logger.remove(self.handlers["stderr"])
            self.handlers["stderr"] = logger.add(
                sys.stderr,
                format="<level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                colorize=True,
                level=newLevel.upper()
            )
        return logger

class CFG:
    _instance = None
    _initialized = False

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None: cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if not self._initialized:
            self._initialized = True
            self.config_path = config_path or Path(__file__).parent / "config.yaml"
            self._config = None
            self.user_dir = Path.home() / ".config" / "llm4dd"
            self.backup_dir = self.user_dir / "config_backups"
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            self.reload()
    
    def reload(self) -> None:
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.debug(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._config = {}
    
    def get(self, path: str, default:Any=None) -> Any:
        try:
            value = self._config
            for key in path.split('.'): value = value[key]
            return value
        except (KeyError, TypeError): return default
    
    def set(self, path: str, value: Any, save: bool = True) -> None:
        try:
            config = self._config
            keys = path.split('.')
            for key in keys[:-1]:
                config = config.setdefault(key, {})
            
            config[keys[-1]] = value
            
            if save:
                with open(self.config_path, 'w') as f:
                    yaml.safe_dump(self._config, f, default_flow_style=False)
                logger.debug(f"Configuration saved to {self.config_path}")
                
        except Exception as e:
            logger.error(f"Failed to set configuration value: {e}")
    
    def get_all(self) -> dict: return self._config or {}
        
    def resolve_env_vars(self, value: Any) -> Any:
        if isinstance(value, str):
            import re
            import os
            pattern = r'\${([^}]+)}'
            matches = re.finditer(pattern, value)
            resolved = value
            for match in matches:
                env_var = match.group(1)
                env_value = os.getenv(env_var)
                if env_value is None:
                    logger.warning(f"Environment variable {env_var} not found")
                    continue
                resolved = resolved.replace(f"${{{env_var}}}", env_value)
            return resolved
        elif isinstance(value, dict):
            return {k: self.resolve_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.resolve_env_vars(v) for v in value]
        return value

    def get_path(self, path_key: str) -> Optional[Path]:
        value = self.get(path_key)
        if value is None: return None

        value = self.resolve_env_vars(value)
        
        if isinstance(value, str):
            for k, v in self._config.get('app', {}).get('paths', {}).items():
                if isinstance(v, str) and v.startswith('&'):
                    anchor = v[1:]  # Remove &
                    if anchor in value:
                        anchor_value = self.get(f'app.paths.{anchor}')
                        if anchor_value: value = value.replace(f"&{anchor}", anchor_value)
        
        try: return Path(value).expanduser().resolve()
        except Exception as e:
            logger.error(f"Failed to resolve path {value}: {e}")
            return None

    def backup(self, name: Optional[str] = None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{name}_{timestamp}" if name else timestamp
        backup_path = self.backup_dir / f"config_{backup_name}.yaml"
        
        try:
            shutil.copy2(self.config_path, backup_path)
            os.chmod(backup_path, 0o600)
            logger.info(f"Config backup created at {backup_path}")
        except Exception as e: logger.error(f"Failed to create config backup: {e}")

class Secrets: 
    def __init__(self): 
        self.user_dir = Path.home() / ".config" / "llm4dd"
        self.keys_file = self.user_dir / "keys.json"
        self.user_dir.mkdir(parents=True, exist_ok=True)

    def write(self, data: dict):
        tmp = self.keys_file.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f: json.dump(data, f, indent=2)
        tmp.replace(self.keys_file)
        try: os.chmod(self.keys_file, 0o600)
        except Exception: pass

    def load(self) -> dict: 
        if not self.keys_file.exists(): return {}
        try: 
            with self.keys_file.open("r", encoding="utf-8") as f: return json.load(f) or {}
        except Exception: return {}

    def set(self, name: str, value: str):
        keys = self.load()
        keys[name] = value
        self.write(keys)
    
    def get(self, name: str, fallback_env: Optional[str]=None) -> Optional[str]: 
        keys = self.load()
        v = keys.get(name)
        if v: return v
        if fallback_env: return os.getenv(fallback_env)
        return None
    
    def list(self) -> list: return list(self.load().keys())

    def remove(self, name: str): 
        keys = self.load()
        if name in keys: 
            keys.pop(name)
            self.write(keys)

if __name__ == "__main__":
    cfg = CFG()
    log = Log()
    sec = Secrets()
    parser = argparse.ArgumentParser(prog='l4d', description='Command line interface for LLM4DD')

    parser.add_argument('--verbose', '-v', 
        type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        default="INFO",
        help="Choose verbosity level."
    )
    parser.add_argument('--api-keys', '-k',
        type=str, required=True, choices=['set', 'load'], 
        help="Configure API keys. [load] pulls keys from '~/.config/llm4dd/keys.json'. [set] allows you to configure them through the command line."
    )
    parser.add_argument('--source-path', '-sp',
        type=str, required=False, default="LOCAL",
        help="Configure path to source data. Defaults to the local path at 'data/'."
    )

    args = parser.parse_args()
    if args.verbose:
        log.changeVerbose(args.verbose)
        logger.info(f"Verbosity level set to {args.verbose}")
    
    if args.api_keys == 'load':
        keys = sec.load()
        if keys:
            logger.info("API keys loaded successfully")
            for key in sec.list(): logger.debug(f"keys found: {sec.list()}")
        else: logger.warning("No API keys found in configuration")
    elif args.api_keys == 'set':
        accepted = ['openai', 'anthropic', 'xai', 'huggingface', 'openrouter', 'novita-ai']
        print("Enter 'q' at any time to finish entering API keys..")
        keys_set = 0
        while True: 
            provider = input("Enter LLM API provider: ").lower().strip()
            if provider=='q': break
            elif provider not in accepted: print(f'Invalid provider. Please enter API keys from the supported providers: {accepted}')
            key = getpass.getpass(prompt="Enter API key: ").strip()
            if key.lower()=='q': break
            sec.set(name=f'{provider.upper()}_API_KEY', value=key); keys_set+=1
        if keys_set==0: raise Exception(f'No keys set..')
        else: logger.info(f'Succesfully set LLM API keys..')

    if str(args.source_path).lower() != "local":
        if not Path(args.source_path).exists(): raise FileExistsError(f"file not found:{args.source_path}")
        cfg.set('env.source_path', args.source_path)
        logger.info(f"Source path set to: {args.source_path}")
    else:
        default_path = Path(__file__).parent / "data"
        if not default_path.exists():
            default_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created default data directory: {default_path}")
        cfg.set('env.source_path', str(default_path))
        logger.info(f"Using default source path: {default_path}")