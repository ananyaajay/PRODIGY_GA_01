Fine-tune GPT-2 on a custom text dataset to generate contextually relevant and stylistically consistent text.
This script uses Hugging Face's Transformers and Datasets libraries.

Features:
- Loads GPT-2 and tokenizer
- Loads your custom text data from a .txt file (one example per line)
- Trains and evaluates the model
- Saves the fine-tuned model

Key Concepts & Standouts:
- Utilizes transfer learning by fine-tuning a pre-trained transformer (GPT-2)
- Handles tokenization and sequence truncation/padding
- Allows for prompt-based generation with stylistic mimicry of your training data
- Easily adaptable to larger GPT models or alternative datasets

Requirements:
- transformers
- datasets
- torch



