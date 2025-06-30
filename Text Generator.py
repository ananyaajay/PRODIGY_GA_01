import argparse
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to training .txt file (one example per line).")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to eval/validation .txt file.")
    parser.add_argument("--output_dir", type=str, default="./gpt2-finetuned", help="Where to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()

def main():
    args = parse_args()
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token # GPT-2 has no pad token by default
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Load datasets
    train_data = load_dataset("text", data_files={"train": args.train_file})
    eval_data = load_dataset("text", data_files={"validation": args.eval_file})

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length"
        )

    tokenized_train = train_data["train"].map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_eval = eval_data["validation"].map(tokenize_function, batched=True, remove_columns=["text"])

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train and save
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")

    # Optional: Generate a sample
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=80,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )
    print("\nSample Generation:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
