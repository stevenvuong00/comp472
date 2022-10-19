import numpy as np
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from DatasetPreparation import json_load, emotionsList, sentiments_list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier


# 2.1
# Load and Process Data
def process_data(print_vocab=False):
    vec = CountVectorizer()
    commentsList = [list[0] for list in json_load]  # list all the first elements, which is the sentence
    commentsVectorized = vec.fit_transform(commentsList)  # encoded list of list of strings
    emotions = [list[1] for list in json_load]  # list of all emotions
    sentiments = [list[2] for list in json_load]  # list of all sentiments

    if print_vocab:
        vocabulary = vec.get_feature_names_out()
        vocab_count = np.asarray(commentsVectorized.sum(axis=0))[0]
        f = open("outputs/2-1/vocabulary.txt", "w", encoding="utf-8")
        lexicon = dict(zip(vocabulary, vocab_count))
        for word, count in lexicon.items():
            f.write(word + " : " + str(count) + "\n")

        f.close()

    return commentsVectorized, emotions, sentiments


# 2.2
# Splitting the dataset
def split_dataset_emotion():
    commentsVectorized, emotions, sentiments = process_data()
    x_emotion_training, x_emotion_test, y_emotion_training, y_emotion_test = train_test_split(commentsVectorized,
                                                                                              emotions, test_size=0.20,
                                                                                              random_state=77)
    return x_emotion_training, x_emotion_test, y_emotion_training, y_emotion_test


def split_dataset_sentiment():
    commentsVectorized, emotions, sentiments = process_data()
    x_sentiment_training, x_sentiment_test, y_sentiment_training, y_sentiment_test = train_test_split(
        commentsVectorized, sentiments, test_size=0.20, random_state=77)
    return x_sentiment_training, x_sentiment_test, y_sentiment_training, y_sentiment_test


x_emotion_training, x_emotion_test, y_emotion_training, y_emotion_test = split_dataset_emotion()
x_sentiment_training, x_sentiment_test, y_sentiment_training, y_sentiment_test = split_dataset_sentiment()


# 2.3.1
def base_mnb():
    # Classifying the data
    classifier = MultinomialNB()
    nb_emotion = classifier.fit(x_emotion_training, y_emotion_training)

    # Testing the model
    y_emotion_pred = nb_emotion.predict(x_emotion_test)

    # file open
    fe = open("outputs/2-3/Base-MNB_Emotion.txt", "w")

    fe.write("Base Emotions Naive Bayes Model")
    np.savetxt(fe, confusion_matrix(y_emotion_test, y_emotion_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fe.write(classification_report(y_emotion_test, y_emotion_pred, digits=5))

    fe.close()

    # # Redoing the same thing for sentiments
    nb_sentiment = classifier.fit(x_sentiment_training, y_sentiment_training)

    # # Testing the model
    y_sentiment_pred = nb_sentiment.predict(x_sentiment_test)

    fs = open("outputs/2-3/Base-MNB_Sentiment.txt", "w")

    fs.write("Base Sentiments Naive Bayes Model")
    np.savetxt(fs, confusion_matrix(y_sentiment_test, y_sentiment_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fs.write(classification_report(y_sentiment_test, y_sentiment_pred, digits=5))
    fs.close()


# 2.3.2
def base_dt():
    # Classifying the data
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(x_emotion_training, y_emotion_training)

    # Testing the model
    y_emotion_pred = classifier.predict(x_emotion_test)

    # Getting output
    fe = open("outputs/2-3/Base-DT_Emotion.txt", "w")

    fe.write("Base Emotions Decision Tree Model")
    np.savetxt(fe, confusion_matrix(y_emotion_test, y_emotion_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fe.write(classification_report(y_emotion_test, y_emotion_pred, digits=5))

    fe.close()

    # Redoing the same thing for sentiments
    classifier.fit(x_sentiment_training, y_sentiment_training)

    # Testing the model
    y_sentiment_pred = classifier.predict(x_sentiment_test)

    fs = open("outputs/2-3/Base-DT_Sentiment.txt", "w")

    fs.write("Base Sentiments Decision Tree Model")
    np.savetxt(fs, confusion_matrix(y_sentiment_test, y_sentiment_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fs.write(classification_report(y_sentiment_test, y_sentiment_pred, digits=5))

    fs.close()


def base_mlp():
    # Classifying the data
    classifier = MLPClassifier()
    classifier.fit(x_emotion_training, y_emotion_training)

    # Testing the model
    y_emotion_pred = classifier.predict(x_emotion_test)

    # Getting output
    fe = open("outputs/2-3/Base-MLP_Emotion.txt", "w")

    fe.write("Base Emotions Multi-Layered Perceptron")
    np.savetxt(fe, confusion_matrix(y_emotion_test, y_emotion_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fe.write(classification_report(y_emotion_test, y_emotion_pred, digits=5))

    fe.close()

    # Redoing the same thing for sentiments
    classifier.fit(x_sentiment_training, y_sentiment_training)

    # Testing the model
    y_sentiment_pred = classifier.predict(x_sentiment_test)

    fs = open("outputs/2-3/Base-MLP_Sentiment.txt", "w")

    fs.write("Base Sentiments Multi-Layered Perceptron")
    np.savetxt(fs, confusion_matrix(y_sentiment_test, y_sentiment_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fs.write(classification_report(y_sentiment_test, y_sentiment_pred, digits=5))

    fs.close()


# 2.3.4
def top_mnb():
    # Defining different parameters values we want to test
    params = {
        'alpha': [0, 0.5, 1, 2, 3]
    }

    # Using GridSearchCV to find the best hyper-parameters to use
    mnb = MultinomialNB()
    gs = GridSearchCV(estimator=mnb, param_grid=params)
    gs.fit(x_emotion_training, y_emotion_training)
    best_params = gs.best_params_

    # Applying the best hyper-parameters found by GridSearchCV to our classifier
    improved_mnb = MultinomialNB(alpha=best_params['alpha'])

    # Classifying the data and testing the model
    improved_mnb.fit(x_emotion_training, y_emotion_training)
    improved_mnb_emotion_pred = improved_mnb.predict(x_emotion_test)

    # Getting the output
    fe = open("outputs/2-3/Top-MNB_Emotion.txt", "w")

    fe.write("Top Emotions Naive Bayes Model")
    fe.write("Best hyperparams: {}".format(best_params))
    np.savetxt(fe, confusion_matrix(y_emotion_test, improved_mnb_emotion_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fe.write(classification_report(y_emotion_test, improved_mnb_emotion_pred, digits=5))
    fe.close()

    # Redoing the same thing for sentiments
    # Using GridSearchCV to find the best hyper-parameters to use
    gs.fit(x_sentiment_training, y_sentiment_training)
    best_params = gs.best_params_

    # Applying the best hyper-parameters found by GridSearchCV to our classifier
    improved_mnb = MultinomialNB(alpha=best_params['alpha'])

    # Classifying the data and testing the model
    improved_mnb.fit(x_sentiment_training, y_sentiment_training)
    improved_mnb_sentiment_pred = improved_mnb.predict(x_sentiment_test)

    # Getting the output
    fs = open("outputs/2-3/Top-MNB_Sentiment.txt", "w")

    fs.write("Top Sentiments Naive Bayes Model")
    fs.write("Best hyperparams: {}".format(best_params))
    np.savetxt(fs, confusion_matrix(y_sentiment_test, improved_mnb_sentiment_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fs.write(classification_report(y_sentiment_test, improved_mnb_sentiment_pred, digits=5))

    fs.close()


# 2.3.5
def top_dt():
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [50, 100],
        'min_samples_split': [2, 5, 8]
    }

    dtc = tree.DecisionTreeClassifier()

    gs = GridSearchCV(estimator=dtc, param_grid=params)

    gs.fit(x_emotion_training, y_emotion_training)

    best_params = gs.best_params_  # {'criterion': 'gini', 'max_depth': 50, 'min_samples_split': 8}

    improved_dtc = tree.DecisionTreeClassifier(criterion=best_params['criterion'], max_depth=best_params['max_depth'],
                                               min_samples_split=best_params['min_samples_split'])
    improved_dtc.fit(x_emotion_training, y_emotion_training)
    improved_dtc_emotion_pred = improved_dtc.predict(x_emotion_test)

    fe = open("outputs/2-3/Top-DT_Emotion.txt", "w")

    fe.write("Top Emotions Decision Tree Model")
    fe.write("Best hyperparams: {}".format(best_params))
    np.savetxt(fe, confusion_matrix(y_emotion_test, improved_dtc_emotion_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fe.write(classification_report(y_emotion_test, improved_dtc_emotion_pred, digits=5))

    fe.close()

    gs.fit(x_sentiment_training, y_sentiment_training)
    best_params == gs.best_params_

    improved_dtc = tree.DecisionTreeClassifier(criterion=best_params['criterion'], max_depth=best_params['max_depth'],
                                               min_samples_split=best_params['min_samples_split'])
    improved_dtc.fit(x_sentiment_training, y_sentiment_training)
    improved_dtc_sentiment_pred = improved_dtc.predict(x_sentiment_test)

    fs = open("outputs/2-3/Top-DT_Sentiment.txt", "w")

    fs.write("Top Sentiments Decision Tree Model")
    fs.write("Best hyperparams: {}".format(best_params))
    np.savetxt(fs, confusion_matrix(y_sentiment_test, improved_dtc_sentiment_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fs.write(classification_report(y_sentiment_test, improved_dtc_sentiment_pred, digits=5))

    fs.close()


def top_MLP():
    print('==================================================')
    print("Top-MLP")
    params = {
        'activation': ['logistic', 'tanh', 'relu', 'identity'],
        'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
        'solver': ['sgd', 'adam']
    }

    # Classifying the data
    mlp = MLPClassifier(max_iter=5)
    gs = GridSearchCV(estimator=mlp, param_grid=params)
    gs.fit(x_emotion_training, y_emotion_training)

    best_params = gs.best_params_

    improved_mlp = MLPClassifier(activation=best_params['activation'],
                                 hidden_layer_sizes=best_params['hidden_layer_sizes'], solver=best_params['solver'],
                                 max_iter=5)
    improved_mlp.fit(x_emotion_training, y_emotion_training)
    improved_mlp_emotion_pred = improved_mlp.predict(x_emotion_test)

    fe = open("outputs/2-3/Top-MLP_Emotion.txt", "w")

    fe.write("Top Emotions Multi-Layered Perceptron")
    fe.write("Best hyperparams: {}".format(best_params))
    np.savetxt(fe, confusion_matrix(y_emotion_test, improved_mlp_emotion_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fe.write(classification_report(y_emotion_test, improved_mlp_emotion_pred, digits=5))

    fe.close()

    # Redoing the same thing for sentiments
    gs.fit(x_sentiment_training, y_sentiment_training)
    best_params == gs.best_params_

    improved_mlp = MLPClassifier(activation=best_params['activation'],
                                 hidden_layer_sizes=best_params['hidden_layer_sizes'], solver=best_params['solver'],
                                 max_iter=5)
    improved_mlp.fit(x_sentiment_training, y_sentiment_training)
    improved_mlp_sentiment_pred = improved_mlp.predict(x_sentiment_test)

    fs = open("outputs/2-3/Top-MLP_Sentiment.txt", "w")

    fs.write("Top Sentiments Multi-Layered Perceptron")
    fs.write("Best hyperparams: {}".format(best_params))
    np.savetxt(fs, confusion_matrix(y_sentiment_test, improved_mlp_sentiment_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fs.write(classification_report(y_sentiment_test, improved_mlp_sentiment_pred, digits=5))

    fs.close()


# print vocab to file
process_data(True)
# base_mnb()
# print("base mnb done!")
# base_dt()
# print("base dt done!")
# base_mlp()
# print("base mlp done!")
# top_mnb()
# print("top mnb done!")
# top_dt()
# print("top dt done!")
# top_MLP()
# print("top mlp done!")

print("everything done!")
