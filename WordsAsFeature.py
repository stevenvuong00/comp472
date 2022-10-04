import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import preprocessing
from DatasetPreparation import json_load, emotionsList, sentimentsList



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
    emotionsY = []
    for i, data in enumerate(training_set):
        e = data[1]
        emotionsY.append(emotionsDict[e])

    # Classifying the data
    classifier = MultinomialNB()
    model = classifier.fit(vec.fit_transform([list[0] for list in training_set]), emotionsY)

    # Testing the model
    testY = []
    for data in test_set:
        e = data[1]
        testY.append(emotionsDict[e])

    score = model.score(vec.transform([list[0] for list in test_set]), testY)
    print(score)

# 2.3.2
def decisionTree():
    # training set is dateset from decision tree tutorial X
    np_array = np.array(training_set)
    X = np_array[:, 0:-2]
    y_emotion = np_array[:, -2]

    # encode
    # need to find max number of features (words) in each data entry? put 100 for now
    le = preprocessing.LabelEncoder()
    for feature in range(1):
        # check the length of the current data entry if feature < than nb of words for that dataset
        # feature will go in range to the max number of features
        # need to handle case for the data entries that have less than that
        # must not go out of range
        X[:, feature] = le.fit_transform(X[:, feature])

    dtc = tree.DecisionTreeClassifier(criterion="entropy")
    dtc.fit(X, y_emotion)
    tree.plot_tree(dtc)

decisionTree()
