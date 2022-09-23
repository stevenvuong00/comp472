from DatasetPreparation import json_load, emotionsList, sentimentsList
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np


commentsList =  [list[0] for list in json_load]
vec = CountVectorizer()
commentsVectorized = vec.fit_transform(commentsList)

# 2.1
vocabulary =  vec.get_feature_names_out()
vocabCount = np.asarray(commentsVectorized.sum(axis=0))[0]
print(dict(zip(vocabulary, vocabCount)))

# 2.2
training_set, test_set = train_test_split(commentsList, test_size=0.20, random_state=77)

emotionsVectorized = vec.fit_transform(emotionsList)
emotions = vec.get_feature_names_out()

emotionsDict = {}
for i, emotion in enumerate(emotions):
    emotionsDict[emotion] = i

print(emotionsDict)
emotionsY = []

for i, data in enumerate(json_load):
    e = data[1]
    emotionsY.append(emotionsDict[e])

classifier = MultinomialNB()
model = classifier.fit(vec.fit_transform(commentsList), emotionsY)

test_comment = vec.transform(np.array(['i love you']))

predict = model.predict(test_comment)
print(predict)