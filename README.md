# PRODIGY_GA_01

# AI TEXT GENERATOR 🤖
This is a simple tool that fine-tunes GPT-2 on a custom text dataset to generate coherent and contextually relevant text.
This script uses Hugging Face's Transformers and Datasets libraries.


## ✓ Features:
- Loads GPT-2 and tokenizer
- Loads custom text data from a .txt file 
- Trains and evaluates the model
- Saves the fine-tuned model

## ✓ Key Concepts & Standouts:
- Utilizes transfer learning by fine-tuning a pre-trained transformer (GPT-2)
- Handles tokenization and sequence truncation/padding
- Allows for prompt-based generation with stylistic mimicry of your training data
- Easily adaptable to larger GPT models or alternative datasets

## ✓ Requirements:
- Transformers
- Datasets
- PyTorch (torch)
