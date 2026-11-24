import argparse
import pandas as pd
import torch
from tqdm import tqdm
import time
import os
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer
from together import Together
from openai import OpenAI
import anthropic

# === Inference Functions ===

def run_local_hf(model_name, prompts):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    responses = []
    for prompt in tqdm(prompts, desc="HF Inference"):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        output = model.generate(input_ids, max_new_tokens=500, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
        response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        responses.append(response)
    return responses

def run_openai(prompt_list, constrained=True, model="gpt-4o"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    responses = []
    for prompt in tqdm(prompt_list, desc="GPT Inference"):
        try:
            content = prompt + ("\nOutput only YTA or NTA." if constrained else "")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=512,
                n=1,
            )
            responses.append(response.choices[0].message.content)
        except Exception as e:
            print(f"[OpenAI Error]: {e}")
            responses.append("")
    return responses

def run_anthropic(prompt_list, model="claude-3-7-sonnet-20250219"):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    responses = []
    for prompt in tqdm(prompt_list, desc="Claude Inference"):
        try:
            message = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256
            )
            text = message.content[0].text if message.content and hasattr(message.content[0], 'text') else ''
            responses.append(text)
        except Exception as e:
            print(f"[Anthropic Error]: {e}")
            responses.append("")
    return responses

def run_gemini(prompt_list, model="gemini-1.5-flash"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model_name=model)
    responses = []
    for prompt in tqdm(prompt_list, desc="Claude Inference"):
        try:
            text = model.generate_content(prompt).text
            responses.append(text)
        except Exception as e:
            print(f"[Gemini Error]: {e}")
            responses.append("")
    return responses

def run_together(prompt_list, model):
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    responses = []
    for prompt in tqdm(prompt_list, desc="Together Inference"):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                n=1,
            )
            responses.append(response.choices[0].message.content)
        except Exception as e:
            print(f"[Together Error]: {e}")
            responses.append("")
    return responses

# === Main ===

def main(model_name, test_data_path, output_file, use_together=False):
    df = pd.read_csv(test_data_path)
    prompts = df["prompt"].tolist()

    model_name_lower = model_name.lower()

    if use_together:
        responses = run_together(prompts, model=model_name)
    elif "gpt" in model_name_lower:
        responses = run_openai(prompts, constrained=False, model=model_name)
    elif "claude" in model_name_lower:
        responses = run_anthropic(prompts, model=model_name)
    elif "gemini" in model_name_lower:
        responses = run_gemini(prompts, model=model_name)
    elif "llama" in model_name_lower or "mistral" in model_name_lower:
        responses = run_local_hf(model_name, prompts)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    df[model_name] = responses
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# === Entry Point ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test CSV file with 'prompt' column")
    parser.add_argument("--output_file", type=str, required=True, help="Where to write the output CSV")
    parser.add_argument("--use_together", action="store_true", help="Use Together AI backend")

    args = parser.parse_args()
    main(args.model_name, args.test_data, args.output_file, args.use_together)
