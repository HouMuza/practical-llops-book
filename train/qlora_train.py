from __future__ import annotations
 
import argparse
 
import mlflow
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
 
from obs.mlflow_config import configure_mlflow
from train.data import load_sft_dataset
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--output-dir", default="./outputs/qlora")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    return p.parse_args()
 
 
def main() -> None:
    args = parse_args()
    configure_mlflow("qwen3-qlora-sft")
    with mlflow.start_run(run_name=f"qlora-r{args.r}"):
        mlflow.log_params(vars(args))
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(
            r=args.r,
            lora_alpha=args.alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        dataset = load_sft_dataset(args.train_jsonl, args.model)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_train_epochs=3,
            learning_rate=2e-4,
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
