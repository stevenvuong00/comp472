import gensim.downloader as api
from matplotlib.font_manager import json_load
from nltk import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from WordsAsFeature import json_load
import numpy as np

#nltk.download('punkt')

def embeddingAsFeature(model_name):
    # 3.1 - Loading model
    model = api.load(model_name)

    # Splitting dataset
    training_set, test_set = train_test_split(np.array(json_load), test_size=0.20, random_state=77)

    # Getting comments list in training set
    x_training_comments = np.array(training_set)[:, 0]
    y_training_emotions = np.array(training_set)[:, 1]
    y_training_sentiments = np.array(training_set)[:, 2]
    x_test_comments = np.array(test_set)[:, 0]
    y_test_emotions = np.array(test_set)[:, 1]
    y_test_sentiments = np.array(test_set)[:, 2]

    # 3.2 - Tokenizing the words 
    x_training_tokenized_comments = [word_tokenize(comment) for comment in x_training_comments]
    x_training_tokenized_words = [item for sublist in x_training_tokenized_comments for item in sublist]
    x_test_tokenized_comments = [word_tokenize(words) for words in x_test_comments]
    x_test_tokenized_words = [item for sublist in x_test_tokenized_comments for item in sublist]
    print("Number of tokens in the training set: {}".format(len(x_training_tokenized_words)))

    # 3.3 - Computing the average embeddings
    x_embedded_comments_training = [model.get_mean_vector(tokenized_comment) for tokenized_comment in x_training_tokenized_comments] 
    x_embedded_comments_test = [model.get_mean_vector(tokenized_comment) for tokenized_comment in x_test_tokenized_comments] 

    # 3.4 - Computing hit rates
    training_hit = 0
    test_hit = 0
    for word in x_training_tokenized_words:
        if model.__contains__(word):
            training_hit = training_hit + 1

    for word in x_test_tokenized_words:
        if model.__contains__(word):
            test_hit = test_hit + 1
    print("Training set hit rate: {}".format(training_hit/len(x_training_tokenized_words)))
    print("Test set hit rate: {}".format(test_hit/len(x_test_tokenized_words)))
    
    # 3.5 - training a Base-MLP with default parameters
    base_mlp(x_embedded_comments_training, y_training_emotions, x_embedded_comments_test, y_test_emotions, y_training_sentiments, y_test_sentiments, model_name)

    # 3.6 - training a Top-MLP with chosen hyper-parameters
    top_mlp(x_embedded_comments_training, y_training_emotions, x_embedded_comments_test, y_test_emotions, y_training_sentiments, y_test_sentiments, model_name)


def base_mlp(x_embedded_comments_training, y_training_emotions, x_embedded_comments_test, y_test_emotions, y_training_sentiments, y_test_sentiments, model_name):
    clf = MLPClassifier(max_iter=1)
    clf.fit(x_embedded_comments_training, y_training_emotions)

    # Testing the model
    y_emotion_pred = clf.predict(x_embedded_comments_test)
    
    # file open
    file_name = "[" + model_name + "] Base-MLP_Emotion.txt"
    fe = open("outputs/3-5/" + file_name, "w")

    fe.write("Base Emotions Multi-Layered Perceptron")
    np.savetxt(fe, confusion_matrix(y_test_emotions, y_emotion_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fe.write(classification_report(y_test_emotions, y_emotion_pred, digits=5))
    fe.close()

    # Redoing the same thing for sentiments
    clf.fit(x_embedded_comments_training, y_training_sentiments)

    # Testing the model
    y_sentiment_pred = clf.predict(x_embedded_comments_test)
    
    # file open
    file_name = "[" + model_name + "] Base-MLP_Sentiment.txt"
    fs = open("outputs/3-5/" + file_name, "w")

    fs.write("Base Sentiments Multi-Layered Perceptron")
    np.savetxt(fs, confusion_matrix(y_test_sentiments, y_sentiment_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fs.write(classification_report(y_test_sentiments, y_sentiment_pred, digits=5))
    fs.close()

def top_mlp(x_embedded_comments_training, y_training_emotions, x_embedded_comments_test, y_test_emotions, y_training_sentiments, y_test_sentiments, model_name):
    hyperparams = {'activation': 'logistic', 'hidden_layer_sizes': (30, 50), 'solver': 'adam'}
    clf = MLPClassifier(activation=hyperparams['activation'], hidden_layer_sizes=hyperparams['hidden_layer_sizes'], solver=hyperparams['solver'])

    clf.fit(x_embedded_comments_training, y_training_emotions)

    # Testing the model
    y_emotion_pred = clf.predict(x_embedded_comments_test)
    
    # file open
    file_name = "[" + model_name + "] Top-MLP_Emotion.txt"
    fe = open("outputs/3-5/" + file_name, "w")

    fe.write("Base Emotions Multi-Layered Perceptron")
    np.savetxt(fe, confusion_matrix(y_test_emotions, y_emotion_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fe.write(classification_report(y_test_emotions, y_emotion_pred, digits=5))
    fe.close()

    # Redoing the same thing for sentiments
    clf.fit(x_embedded_comments_training, y_training_sentiments)

    # Testing the model
    y_sentiment_pred = clf.predict(x_embedded_comments_test)
    
    # file open
    file_name = "[" + model_name + "] Top-MLP_Sentiment.txt"
    fs = open("outputs/3-5/" + file_name, "w")

    fs.write("Base Sentiments Multi-Layered Perceptron")
    np.savetxt(fs, confusion_matrix(y_test_sentiments, y_sentiment_pred),
               fmt="%6.1d",
               delimiter=" ",
               header="\nConfusion Matrix",
               footer="===================================\n")
    fs.write(classification_report(y_test_sentiments, y_sentiment_pred, digits=5))
    fs.close()
    

embeddingAsFeature('word2vec-google-news-300')
embeddingAsFeature('fasttext-wiki-news-subwords-300')
embeddingAsFeature('glove-wiki-gigaword-300')
