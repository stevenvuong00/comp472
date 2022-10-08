import numpy as np
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import preprocessing
from DatasetPreparation import json_load, emotionsList, sentimentsList

# 2.1
commentsList = [list[0] for list in json_load]
vec = CountVectorizer()

commentsVectorized = vec.fit_transform(commentsList)
vocabulary = vec.get_feature_names_out()
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
    targets_emotion = [element[1] for element in training_set]
    targets_sentiment = [element[2] for element in training_set]

    # list of only the words without label
    only_words = [element[0] for element in training_set]

    # filter punctuation
    clean_words = []
    for element in only_words:
        clean = element.translate(str.maketrans('', '', string.punctuation))
        clean_words.append(clean)

    # print(len(clean_words))

    # transform each sublist into a list of list
    post_list = [list[0] for list in training_set]
    words_list = [element.split() for element in post_list]

    post_list_all = [list[0] for list in json_load]
    words_list_all = [element.split() for element in post_list_all]

    # max length of all the sublists
    maxLenSublist = max(words_list_all, key=len)
    max_length = len(maxLenSublist) + 1

    padded_words = np.zeros((len(training_set), max_length), dtype='<U25')

    # adding all the words to the new array
    for i in range(len(training_set)):
        for j in range(len(words_list[i])):
            # need to prevent index uot of bound, depends on the length of the orinigal word list
            padded_words[i][j] = words_list[i][j]
    # print(padded_words)

    # re-adding the emotion and sentiments
    dataset = padded_words.copy().tolist()
    for i in range(len(dataset)):
        dataset[i].append(targets_emotion[i])
        dataset[i].append(targets_sentiment[i])

    for i in range(len(words_list)):
        words_list[i].append(targets_emotion[i])
        words_list[i].append(targets_sentiment[i])

    # training set is dateset from decision tree tutorial X
    np_array = np.array(dataset)
    np_array2 = np.array(words_list, dtype=object)
    X = np_array[:, 0:-2]
    y_emotion = np_array[:, -2]

    # encode
    # need to find max number of features (words) in each data entry? put 100 for now
    le = preprocessing.LabelEncoder()
    for feature in range(max_length):
        # check the length of the current data entry if feature < than nb of words for that dataset
        # feature will go in range to the max number of features
        # need to handle case for the data entries that have less than that
        # must not go out of range
        X[:, feature] = le.fit_transform(X[:, feature])

    dtc = tree.DecisionTreeClassifier(criterion="entropy")
    dtc.fit(X, y_emotion)
    print("done!")
    # tree.plot_tree(dtc)
    # print("tree plotted!")

    test_posts = [element[0] for element in test_set]
    test_words = [element.split() for element in test_posts]
    # X_test = np.array(test_words)
    for feature in range(max_length):
        # check the length of the current data entry if feature < than nb of words for that dataset
        # feature will go in range to the max number of features
        # need to handle case for the data entries that have less than that
        # must not go out of range
        # X_test[:, feature] = le.fit_transform(X_test[:, feature])
        test_words[feature] = le.fit_transform(test_words[feature])

    result = dtc.predict(test_words)
    print(dict(result))

    # dtc.predict(training_set)
    # plt = tree.plot_tree(dtc)
    # tree.plot_tree(dtc)
    # print("done")
    # plt.show()

    # dot_data = tree.export_graphviz(dtc, out_file=None,
    # feature_names= clean_words,
    # class_names = targets_emotion,
    # filled =True, rounded=True)
    # graph = graphviz.Source(dot_data)
    # graph.render("mytree")


# decisionTree()

def testxd():
    # get the data
    all_posts = [element[0] for element in json_load]
    max_post_length = 0
    for post in all_posts:
        if len(post.split()) > max_post_length:
            max_post_length = len(post.split())

    training_posts = [element[0] for element in training_set]
    training_target_out = [element[1] for element in training_set]
    training_posts_as_words = [post.split() for post in training_posts]

    padded_words = np.zeros((len(training_set), max_post_length), dtype='<U999') # what is the longest word a redditor can type :)

    # make lexicon of words + turn them lowercase
    label_lexicon = []
    for post in training_posts_as_words:
        for i in range(len(post)):
            post[i] = post[i].lower()
            label_lexicon.append(post[i])

    label_lexicon.append('') # empty string is missing????????????????????

    label_lexicon = np.unique(label_lexicon)
    ordinal_lexicon = []
    for i in range(len(label_lexicon)):
        ordinal_lexicon.append([label_lexicon[i], i])

    for i in range(len(training_set)):
        for j in range(len(training_posts_as_words[i])):
            padded_words[i][j] = training_posts_as_words[i][j]

    print("debug print breakpoint 2")

    enc = preprocessing.OrdinalEncoder()
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    enc.fit(X)

    encoder = preprocessing.LabelEncoder()
    encoder.fit(np.array(label_lexicon))

    experimental_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
    experimental_encoder.fit(ordinal_lexicon)
    print("debug print breakpoint 3 - finished fitting")

    #padded_words[:, 0] = experimental_encoder.transform(padded_words[:, 0])

    for i in range(max_post_length):
        padded_words[:, i] = encoder.transform(padded_words[:, i])
    print("debug print breakpoint 4 - finished transforming")

    dtc = tree.DecisionTreeClassifier(criterion="entropy")
    dtc.fit(padded_words, training_target_out)

    tree.plot_tree(dtc)

    print("debug print breakpoint 5 - done with all tasks")


testxd()
