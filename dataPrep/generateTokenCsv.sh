#!/bin/bash
python dataPrep/tokenizer.py --input "data/IMDB Dataset.csv" --text_column review --output dataPrep/data/movies_tokenized.csv