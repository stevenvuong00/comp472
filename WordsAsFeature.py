from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from DatasetPreparation import json_load, emotionsList, sentimentsList
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree, preprocessing
import numpy as np



# 2.1
commentsList =  [list[0] for list in json_load]
vec = CountVectorizer()

commentsVectorized = vec.fit_transform(commentsList)
vocabulary =  vec.get_feature_names_out()
vocabCount = np.asarray(commentsVectorized.sum(axis=0))[0]
# print(dict(zip(vocabulary, vocabCount)))

# 2.2
# Splitting the dataset
training_set, test_set = train_test_split(json_load, test_size=0.20, random_state=77)

# 2.3.1
def naiveBayes():
    # Extracting the emotions
    vec.fit_transform(emotionsList)
    emotions = vec.get_feature_names_out()

    # Associating emotions to a number
    emotionsDict = {}
    for i, emotion in enumerate(emotions):
        emotionsDict[emotion] = i
    print(emotionsDict)

    # Generating array Y
    yEmotions = []
    for i, data in enumerate(training_set):
        e = data[1]
        yEmotions.append(emotionsDict[e])

    # Classifying the data
    classifier = MultinomialNB()
    model = classifier.fit(vec.fit_transform([list[0] for list in training_set]), yEmotions)

    # Testing the model
    testYEmotions = []
    for data in test_set:
        e = data[1]
        testYEmotions.append(emotionsDict[e])

    scoreEmotions = model.score(vec.transform([list[0] for list in test_set]), testYEmotions)
    print('Emotion score: {}'.format(scoreEmotions))

    # Redoing the same thing for sentiments

    #Extracting the sentiments
    vec.fit_transform(sentimentsList)
    sentiments = vec.get_feature_names_out()

    # Associating sentiments to a number
    sentimentsDict = {}
    for i, sentiment in enumerate(sentiments):
        sentimentsDict[sentiment] = i
    print(sentimentsDict)

    # Generating array Y
    ySentiments = []
    for i, data in enumerate(training_set):
        s = data[2]
        ySentiments.append(sentimentsDict[s])

    # Classifying the data
    model = classifier.fit(vec.fit_transform([list[0] for list in training_set]), ySentiments)

    # Testing the model
    testYSentiments = []
    for data in test_set:
        s = data[2]
        testYSentiments.append(sentimentsDict[s])
    
    scoreSentiments = model.score(vec.transform([list[0] for list in test_set]), testYSentiments)
    print('Sentiments score: {}'.format(scoreSentiments))


def decisionTree():
    encoded = vec.fit_transform(element[0] for element in json_load)
    emotions = [emotion[1] for emotion in json_load]
    sentiments = [sentiment[2] for sentiment in json_load]
    xEmotion_training, xEmotion_test, yEmotion_training, yEmotion_test = train_test_split(encoded, emotions, test_size=0.20, random_state=77)
    xSentiment_training, xSentiment_test, ySentiment_training, ySentiment_test = train_test_split(encoded, sentiments, test_size=0.20, random_state=77)

    dtc = tree.DecisionTreeClassifier(criterion = 'entropy')

    dtc.fit(xEmotion_training, yEmotion_training)
    yEmotion_pred = dtc.predict(xEmotion_test)

    dtc.fit(xSentiment_training, ySentiment_training)
    ySentiment_pred = dtc.predict(xSentiment_test)

    print(classification_report(yEmotion_test, yEmotion_pred))
    print(classification_report(ySentiment_test, ySentiment_pred))

decisionTree()