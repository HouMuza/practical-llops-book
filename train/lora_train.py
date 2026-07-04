from __future__ import annotations
 
import argparse
import logging
 
import mlflow
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
 
from obs.mlflow_config import configure_mlflow
from train.data import load_sft_dataset
 
log = logging.getLogger(__name__)
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--output-dir", default="./outputs/lora")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()
 
 
def main() -> None:
    args = parse_args()
    configure_mlflow("qwen3-lora-sft")
    with mlflow.start_run(run_name=f"lora-r{args.r}"):
        mlflow.log_params(vars(args))
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        peft_config = LoraConfig(
            r=args.r,
            lora_alpha=args.alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        dataset = load_sft_dataset(args.train_jsonl, args.model)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to=["mlflow"],
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=4096,
            args=training_args,
        )
        trainer.train()
        trainer.save_model(args.output_dir)
        mlflow.log_artifacts(args.output_dir, artifact_path="adapter")
 
 
if __name__ == "__main__":
    main()
