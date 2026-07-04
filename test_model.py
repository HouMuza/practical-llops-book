#!/usr/bin/env python3
"""Quick test to verify the downloaded model works"""

from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')

print("\n" + "="*60)
print("Testing model inference...")
print("="*60 + "\n")

prompt = "Explain KV cache in one paragraph."
print(f"Prompt: {prompt}\n")

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=80)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Response:\n{response}\n")
print("="*60)
print("✓ Model is working correctly!")
print("="*60)
