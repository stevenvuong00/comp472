import numpy as np
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from DatasetPreparation import json_load, emotionsList, sentiments_list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier


# 2.1
vec = CountVectorizer()
commentsList =  [list[0] for list in json_load]         # list all the first elements, which is the sentence
commentsVectorized = vec.fit_transform(commentsList)    # encoded list of list of strings
emotions = [list[1] for list in json_load]              # list of all emotions
sentiments = [list[2] for list in json_load]            # list of all sentiments
vocabulary =  vec.get_feature_names_out()
vocabCount = np.asarray(commentsVectorized.sum(axis=0))[0]
# print(dict(zip(vocabulary, vocabCount)))

# 2.2
# Splitting the dataset
def split_dataset_emotion():
    x_emotion_training, x_emotion_test, y_emotion_training, y_emotion_test = train_test_split(commentsVectorized, emotions, test_size=0.20, random_state=77)
    return x_emotion_training, x_emotion_test, y_emotion_training, y_emotion_test

x_emotion_training, x_emotion_test, y_emotion_training, y_emotion_test = split_dataset_emotion()

def split_dataset_sentiment():
    x_sentiment_training, x_sentiment_test, y_sentiment_training, y_sentiment_test  = train_test_split(commentsVectorized, sentiments, test_size=0.20, random_state=77)
    return x_sentiment_training, x_sentiment_test, y_sentiment_training, y_sentiment_test 

# 2.3.1
def naive_bayes():

    # Classifying the data
    classifier = MultinomialNB()
    nb_emotion = classifier.fit(x_emotion_training, y_emotion_training)

    # Testing the model
    y_emotion_pred = nb_emotion.predict(x_emotion_test)

    print("Emotions Naive Bayes Model")
    print(confusion_matrix(y_emotion_test, y_emotion_pred))
    print(classification_report(y_emotion_test, y_emotion_pred))

    emotion_score = classifier.score(x_emotion_test, y_emotion_test)
    print("emotion_score")
    print(emotion_score)

    # Redoing the same thing for sentiments
    nb_sentiment = classifier.fit(x_emotion_training, y_emotion_training)

    # Testing the model
    y_emotion_pred = nb_sentiment.predict(x_sentiment_test)

    print("Sentiments Naive Bayes Model")
    print(confusion_matrix(y_emotion_test, y_emotion_pred))
    print(classification_report(y_emotion_test, y_emotion_pred))

    sentiment_score = classifier.score(x_sentiment_test, y_sentiment_test)
    print("sentiment_score")
    print(sentiment_score)

# 2.3.2
def decision_tree():
    # Classifying the data
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(x_emotion_training, y_emotion_training)

    # Testing the model
    y_emotion_pred = classifier.predict(x_emotion_test)

    # Getting output
    print("Emotions Decision Tree Model")
    print(confusion_matrix(y_emotion_test, y_emotion_pred))
    print(classification_report(y_emotion_test, y_emotion_pred))

    emotion_score = classifier.score(x_emotion_test, y_emotion_test)
    print("emotion_score")
    print(emotion_score)


    # Redoing the same thing for sentiments
    classifier.fit(x_sentiment_training, y_sentiment_training)

    # Testing the model
    y_sentiment_pred = classifier.predict(x_sentiment_test)

    print("Sentiments Decision Tree Model")
    print(confusion_matrix(y_sentiment_test, y_sentiment_pred))
    print(classification_report(y_sentiment_test, y_sentiment_pred))

    sentiment_score = classifier.score(x_sentiment_test, y_sentiment_test)
    print("sentiment_score")
    print(sentiment_score)


def multi_layered_perceptron():
    # Classifying the data
    classifier = MLPClassifier()
    classifier.fit(x_emotion_training, y_emotion_training)

    # Testing the model
    y_emotion_pred = classifier.predict(x_emotion_test)

    # Getting output
    print("Emotions Multi-Layered Perceptron")
    print(confusion_matrix(y_emotion_test, y_emotion_pred))
    print(classification_report(y_emotion_test, y_emotion_pred))

    emotion_score = classifier.score(x_emotion_test, y_emotion_test)
    print("emotion_score")
    print(emotion_score)

    # Redoing the same thing for sentiments
    classifier.fit(x_sentiment_training, y_sentiment_training)

    # Testing the model
    y_sentiment_pred = classifier.predict(x_sentiment_test)

    print("Sentiments Decision Tree Model")
    print(confusion_matrix(y_sentiment_test, y_sentiment_pred))
    print(classification_report(y_sentiment_test, y_sentiment_pred))

    sentiment_score = classifier.score(x_sentiment_test, y_sentiment_test)
    print("sentiment_score")
    print(sentiment_score)

x_sentiment_training, x_sentiment_test, y_sentiment_training, y_sentiment_test  = split_dataset_sentiment()
# naive_bayes()
decision_tree()
# multi_layered_perceptron()
