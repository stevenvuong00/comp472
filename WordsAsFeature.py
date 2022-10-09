from sklearn import preprocessing
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


# def decisionTree():
#     X = np.array(training_set)[:, 0]
#     print(X)
#     yEmotions = np.array(training_set)[:, 1]
#     # print(yEmotions)
#     le = preprocessing.LabelEncoder()
#     X = le.fit_transform(X)
#     print(X)
#     dtc = tree.DecisionTreeClassifier(criterion="entropy")
#     dtc.fit(X, yEmotions)
#     # tree.plot_tree(dtc)


naiveBayes()
# decisionTree()