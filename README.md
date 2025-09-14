# LLM4DD 

*Status: In Development*

## Description

This project implements a hybridized pipeline that integrates large language models (LLMs) with cheminformatics, machine learning, and neural networks to identify and characterize novel inhibitors of chosen biological targets. The pipeline unifies empirical screening data, molecular descriptors, and interpretable rule-based reasoning into a modular, multi-phase architecture inspired by LLM4SD.

<!-- ### Core Dependencies
- **Data Processing**: numpy, polars, scikit-learn, tqdm
- **Cheminformatics**: rdkit, pubchempy
- **LLM Integration**: litellm, transformers, tiktoken, huggingface_hub
- **Deep Learning**: pytorch

### Computational Requirements
- Python 3.8+
- LLM API key: ['openai', 'anthropic', 'xai', 'huggingface', 'openrouter', 'novita-ai'] -->

### Project Framework

This project follows a modular framework for drug discovery, combining data acquisition, preprocessing, feature engineering, dimensionality reduction, and machine learning. The pipeline is designed to:

- Collect and preprocess chemical and biological data from various sources.
- Engineer and reduce features to create meaningful representations of molecular data.
- Train and evaluate machine learning models to predict compound activity and identify key features.
- Provide interpretability and insights into model predictions.
- Enable deployment for inference on new compounds and integration into broader workflows.
