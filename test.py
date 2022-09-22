import numpy as np
import json
import gzip
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

file_path = "goemotions.json.gz"
data = gzip.open(file_path, 'r')
print(data)