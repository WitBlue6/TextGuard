#!/usr/bin/env python3
"""
推理脚本 - 对单条文本进行语法纠错
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(source: str) -> str:
    """格式化提示词"""
    return f"<|im_start|>user\n请帮我纠正以下文本的语法错误：\n{source}\n<|im_end|>\n<|im_start|>assistant\n"


def correct_text(model_path: str, text: str, temperature: float = 0.1, max_new_tokens: int = 128):
    """
    对文本进行语法纠错
    """
    print(f"加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    prompt = format_prompt(text)
    print(f"\n原文: {text}")
    print(f"提示词: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取生成的部分
    if "<|im_end|>assistant" in generated_text:
        corrected = generated_text.split("<|im_end|>assistant")[1].strip()
    else:
        corrected = generated_text.replace(prompt, "").strip()

    print(f"\n修正后: {corrected}")
    return corrected


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="微调后的模型路径")
    parser.add_argument("--text", type=str, required=True, help="待纠正的文本")
    parser.add_argument("--temperature", type=float, default=0.1, help="生成温度")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="最大生成token数")

    args = parser.parse_args()

    correct_text(args.model_path, args.text, args.temperature, args.max_new_tokens)
