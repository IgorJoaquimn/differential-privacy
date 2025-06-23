import argparse
import pandas as pd
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Tokenize a text column in a CSV file.")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--text_column", required=True, help="Name of the text column to tokenize")
    parser.add_argument("--output", required=True, help="Output CSV file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    df = pd.read_csv(args.input, low_memory=False)

    if args.text_column not in df.columns:
        raise ValueError(f"Column {args.text_column} not found in input file.")

    def tokenize_text(text):
        return tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=512
        )


    df[f"{args.text_column}_tokens"] = df[args.text_column].astype(str).apply(tokenize_text)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
