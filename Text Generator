import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Path to your custom training text file")
    args = parser.parse_args()

    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=args.dataset,
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    training_args = TrainingArguments(
        output_dir='./gpt2-finetuned',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model('./gpt2-finetuned')
    tokenizer.save_pretrained('./gpt2-finetuned')

if __name__ == '__main__':
    main()
