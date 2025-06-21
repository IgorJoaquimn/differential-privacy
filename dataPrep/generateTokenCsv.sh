#!/bin/bash
python dataPrep/tokenizer.py --input dataPrep/data/movies_metadata.csv --text_column overview --output dataPrep/data/movies_tokenized.csv