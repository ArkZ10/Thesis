from openai import OpenAI
import json
from tqdm import tqdm
import pandas as pd
import os
import argparse
from pathlib import Path

def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        try:
            with open('key.txt','r') as f:
                api_key = [line.rstrip('\n') for line in f][0]
        except:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set and no key.txt file to read API key found.")
    return api_key

def format_prompt(text,aita_binary=False):
    if aita_binary:
        return text + "\nOutput only YTA or NTA."
    else:
        return text

def main(args):
    # Load input CSV
    df = pd.read_csv(args.input_file)
    if args.input_column not in df.columns:
        raise ValueError(f"Input column '{args.input_column}' not found in the file.")
    
    # Set up OpenAI
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)

    # Infer default output column and file if not provided
    if args.output_column is None:
        args.output_column = f"{args.input_column}_response"

    if args.output_file is None:
        input_stem = Path(args.input_file).stem
        args.output_file = f"{input_stem}_responses.csv"
    else:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
     # Check if the output file exists and the column is already present
    if os.path.exists(args.output_file):
        output_df = pd.read_csv(args.output_file)
        if args.output_column in output_df.columns:
            raise ValueError(f"Output column '{args.output_column}' already exists in the output file. Choose a different name.")
    
    # Process each row in the input column
    outputs = []
    for text in tqdm(df[args.input_column], desc="Processing rows"):
        prompt = format_prompt(text, args.AITA_binary)
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,
            )
            output = response.choices[0].message.content.strip()
        except Exception as e:
            output = f"[ERROR] {e}"
        outputs.append(output)

    # Save to output column
    df[args.output_column] = outputs
    df.to_csv(args.output_file, index=False)
    print(f"Saved output to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI completion on a CSV column.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--input_column", type=str, required=True, help="Column to read prompts from.")
    parser.add_argument("--output_column", type=str, required=False, help="Column to save responses to.")
    parser.add_argument("--output_file", type=str, required=False, help="Path to the output CSV file.")
    parser.add_argument("--AITA_binary", action="store_true", help="If set, prompts the model to only determine whether the asker is YTA or NTA.")
    args = parser.parse_args()

    main(args)