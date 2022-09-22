<<<<<<< HEAD
import json
import gzip
# import matplotlib.pyplot as plt

f = gzip.open('C:\\Users\\15146\\School\\COMP 472 Labs\\project 1\\comp472\\goemotions.json.gz', 'rb')

json_load = json.load(f)
# jsonArr = np.array(json_load)
# x = [jsonArr[key] for key in jsonArr]
# print(x)
print(json.dumps(json_load, indent=4))

=======
import numpy as np
import json
import gzip
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

file_path = "goemotions.json.gz"
f = gzip.open(file_path, 'rb')

json_load = json.load(f)
print(json.dumps(json_load, indent=4))
>>>>>>> 70eb35c3aaf2517420f4d3c2b80ff83e60553095
