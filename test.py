import numpy as np
import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

file_path = "goemotions.json.gz"
with open(file_path, "r") as data:
    data_set = json.load(data)