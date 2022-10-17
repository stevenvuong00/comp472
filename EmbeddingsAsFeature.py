import gensim.downloader as api
from matplotlib.font_manager import json_load
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from WordsAsFeature import json_load
import numpy as np

# print(len(np.array(json_load)))
#nltk.download('punkt')

# 3.1 - Loading model
model = api.load('word2vec-google-news-300')
# print(model)

# Splitting dataset
training_set, test_set = train_test_split(np.array(json_load), test_size=0.20, random_state=77)

# Getting comments list in training set
training_comments = np.array(training_set)[:, 0]
test_comments = np.array(test_set)[:, 0]

# 3.2 - Tokenizing the words 
training_tokenized_comments = [word_tokenize(comment) for comment in training_comments]
training_tokenized_words = [item for sublist in training_tokenized_comments for item in sublist]
test_tokenized_comments = [word_tokenize(words) for words in test_comments]
test_tokenized_words = [item for sublist in test_tokenized_comments for item in sublist]
print("Number of tokens in the training set: {}".format(len(training_tokenized_words)))

# 3.3 - Computing the average embeddings
# Removing words with no embedding in Word2Vec
for i, comment in enumerate(training_tokenized_comments):
    for j, word in enumerate(comment):
        if(not model.__contains__(word)):
            del training_tokenized_comments[i][j]
for i, comment in enumerate(test_tokenized_comments):
    for j, word in enumerate(comment):
        if(not model.__contains__(word)):
            del test_tokenized_comments[i][j]
training_embedded_comments = [model.get_mean_vector(tokenized_comment) for tokenized_comment in training_tokenized_comments] # Not sure about this
test_embedded_comments = [model.get_mean_vector(tokenized_comment) for tokenized_comment in test_tokenized_comments] # Not sure about this

# 3.4 - Computing hit rates
# training_hit = 0
# test_hit = 0
# for word in training_tokenized_words:
#     if model.__contains__(word):
#         training_hit = training_hit + 1

# for word in test_tokenized_words:
#     if model.__contains__(word):
#         test_hit = test_hit + 1

# accuracy = model.log_accuracy(training_tokenized_words)
# print(accuracy)

# print("Training set hit rate: {}".format(training_hit/len(training_tokenized_words)))
# print("Test set hit rate: {}".format(test_hit/len(test_tokenized_words)))

def base_mlp():
    clf = MLPClassifier()
    training_emotions = [list[1] for list in training_set]
    test_emotions = [list[1] for list in test_set]
    clf.fit(training_embedded_comments, training_emotions)
    score = clf.score(test_embedded_comments, test_emotions)
    # Getting error here
    # ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet. Took about 10mins?
    print(score)

# base_mlp()