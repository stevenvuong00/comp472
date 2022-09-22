from DatasetPreparation import json_load, emotionsList, sentimentsList
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np

wordsList =  [list[0] for list in json_load]

vec = CountVectorizer()
vec_fit = vec.fit_transform(wordsList)

vocabulary =  vec.get_feature_names_out()
vocabCount = np.asarray(vec_fit.sum(axis=0))[0]

print(dict(zip(vocabulary, vocabCount)))