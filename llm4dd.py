#!/usr/bin/env python3 
import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
# from tools import OR, Huggingface
from utils import Log, CFG, Secrets

load_dotenv()

class LLM4DD:
    def __init__(self, provider: str = "openrouter"):
        self.cfg = CFG()
        self.log = Log()
        self.secrets = Secrets()
        self.provider = provider.lower()
        self.client = self._initialize_llm()
        
    def _initialize_llm(self):
        api_key = self.secrets.get(f"{self.provider.upper()}_API_KEY")
        if not api_key:
            raise ValueError(f"No API key found for {self.provider}. Use utils.py to set up API keys.")
            
        if self.provider == "openrouter": return OR(self.cfg, api_key)
        elif self.provider == "huggingface": return Huggingface(api_key)
        else: raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def get_client(self): return self.client

if __name__ == "__main__":
    try:
        llm = LLM4DD()
        client = llm.get_client()
        logger.info(f"Successfully initialized LLM client")
    except Exception as e: logger.error(f"Failed to initialize LLM: {e}")

