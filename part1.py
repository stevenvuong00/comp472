import numpy as np
import json
import gzip
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

file_path = "goemotions.json.gz"
f = gzip.open(file_path, 'rb')

json_load = json.load(f)
print(json.dumps(json_load, indent=4))
