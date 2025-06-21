#!/bin/bash
python tokenizer.py   --input data/movies_metadata.csv   --text_column overview   --tokenizer bert-base-uncased   --output data/movies_tokenized.csv