#!/bin/bash
curl -L -o ./imdb-dataset-of-50k-movie-reviews.zip\
  https://www.kaggle.com/api/v1/datasets/download/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

unzip -o ./imdb-dataset-of-50k-movie-reviews.zip -d ./data
rm ./imdb-dataset-of-50k-movie-reviews.zip
echo "Data downloaded and extracted to ./data"