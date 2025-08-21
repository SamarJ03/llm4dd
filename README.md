# LLM4DD 

*Status: In Development*

## Description

This project implements a hybridized pipeline that integrates large language models (LLMs) with cheminformatics and machine learning methods to identify and characterize novel inhibitors of chosen biological target. The pipeline unifies empirical screening data, molecular descriptors, and interpretable rule-based reasoning into a modular, multi-phase architecture inspired by LLM4SD.

### Core Dependencies
- **Data Processing**: numpy, polars, scikit-learn, tqdm
- **Cheminformatics**: rdkit, pubchempy
- **LLM Integration**: litellm, transformers, tiktoken, huggingface_hub
- **Deep Learning**: pytorch

### Computational Requirements
- Python 3.8+
- LLM API key: ['openai', 'anthropic', 'xai', 'huggingface', 'openrouter', 'novita-ai']
  - openai
  - anthropic
  - xAI
  - hugging face
  - openrouter
  - novita AI

### Biodata Requirments
*...to be specified...*
<!-- Input data must be high-throughput morphological screening results contained in a supported file type. Each compound entry (row) must include a unique identifier, a valid canonical SMILES format of the compound, and normalized 'activity score'.  -->

### Project Structure
```
llm4dd/
├─ data/ 
│  ├─ features/ 
│  │  ├─ meta/ 
│  │  │  ├─ ecfp4_meta.csv
│  │  │  ├─ maccs_meta.csv
│  │  │  └─ rdkit_meta.csv
│  │  ├─ ecfp4.csv 
│  │  ├─ maccs.csv
│  │  └─ rdkit.csv
│  ├─ scaffold/
│  │  ├─ ecfp4/
│  │  ├─ maccs/
│  │  └─ rdkit/ 
│  │     ├─ EState/ 
│  │     ├─ fingerprintBased/
│  │     ├─ functionalGroupCount/
│  │     ├─ molecularTopology/
│  │     ├─ physiochemical/
│  │     ├─ structural/
│  │     ├─ surfaceArea/
│  ├─ features.csv
│  └─ lib.csv
├─ logs/ 
│  ├─ .cache/
│  ├─ debug/
│  └─ error/
├─ resources/
│  └─ results/
├─ src/
│  ├─ CodeGenAndEval.py
│  ├─ CreatePrompts.py
│  ├─ dataPrep.py
│  ├─ Inference.py
│  ├─ SummarizeRules.py
│  └─ Synthesize.py
├─ .gitignore
├─ config.yaml
├─ environment.yml
├─ llm4dd.py
├─ README.md
└─ utils.py
```