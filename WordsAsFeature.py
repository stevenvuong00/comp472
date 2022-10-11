import numpy as np
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from DatasetPreparation import json_load, emotionsList, sentiments_list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier



# 2.1
vec = CountVectorizer()
def extract_feature():
    commentsList =  [list[0] for list in json_load]

    commentsVectorized = vec.fit_transform(commentsList)
    vocabulary =  vec.get_feature_names_out()
    vocabCount = np.asarray(commentsVectorized.sum(axis=0))[0]
    # print(dict(zip(vocabulary, vocabCount)))

# 2.2
# Splitting the dataset
def split_dataset():
    training_set, test_set = train_test_split(json_load, test_size=0.20, random_state=77)
    return training_set, test_set

# 2.3.1
def base_mnb():
    training_set, test_set = split_dataset()

    # Extracting the emotions
    vec.fit_transform(emotionsList)
    emotions = vec.get_feature_names_out()

    # Associating emotions to a number
    emotions_dict = {}
    for i, emotion in enumerate(emotions):
        emotions_dict[emotion] = i
    print(emotions_dict)

    # Generating array Y
    y_emotions = []
    for i, data in enumerate(training_set):
        e = data[1]
        y_emotions.append(emotions_dict[e])

    # Classifying the data
    classifier = MultinomialNB()
    model = classifier.fit(vec.fit_transform([list[0] for list in training_set]), y_emotions)

    # Testing the model
    test_y_emotions = []
    for data in test_set:
        e = data[1]
        test_y_emotions.append(emotions_dict[e])

    y_emotions_pred = classifier.predict(vec.transform([list[0] for list in test_set]))
    print(confusion_matrix(test_y_emotions, y_emotions_pred))
    print(classification_report(test_y_emotions, y_emotions_pred))
    # score_emotions = model.score(vec.transform([list[0] for list in test_set]), test_y_emotions)
    # print('Emotion score: {}'.format(score_emotions))

    # Redoing the same thing for sentiments

    #Extracting the sentiments
    vec.fit_transform(sentiments_list)
    sentiments = vec.get_feature_names_out()

    # Associating sentiments to a number
    sentiments_dict = {}
    for i, sentiment in enumerate(sentiments):
        sentiments_dict[sentiment] = i
    print(sentiments_dict)

    # Generating array Y
    y_sentiments = []
    for i, data in enumerate(training_set):
        s = data[2]
        y_sentiments.append(sentiments_dict[s])

    # Classifying the data
    model = classifier.fit(vec.fit_transform([list[0] for list in training_set]), y_sentiments)

    # Testing the model
    test_y_sentiments = []
    for data in test_set:
        s = data[2]
        test_y_sentiments.append(sentiments_dict[s])

    y_emotions_pred = classifier.predict(vec.transform([list[0] for list in test_set]))
    print(confusion_matrix(test_y_sentiments, y_emotions_pred))
    print(classification_report(test_y_sentiments, y_emotions_pred))
    score_sentiments = model.score(vec.transform([list[0] for list in test_set]), test_y_sentiments)
    print('Sentiments score: {}'.format(score_sentiments))


# 2.3.2
def base_dt():
    print('==================================================')
    print('Base-DT')
    encoded = vec.fit_transform(element[0] for element in json_load)
    emotions = [emotion[1] for emotion in json_load]
    sentiments = [sentiment[2] for sentiment in json_load]

    x_emotion_training, x_emotion_test, y_emotion_training, y_emotion_test = train_test_split(encoded, emotions, test_size=0.20, random_state=77)
    x_sentiment_training, x_sentiment_test, y_sentiment_training, y_sentiment_test = train_test_split(encoded, sentiments, test_size=0.20, random_state=77)

    dtc = tree.DecisionTreeClassifier()

    dtc.fit(x_emotion_training, y_emotion_training)
    y_emotion_pred = dtc.predict(x_emotion_test)

    dtc.fit(x_sentiment_training, y_sentiment_training)
    y_sentiment_pred = dtc.predict(x_sentiment_test)

    print('Emotions (default parameters)')
    print(confusion_matrix(y_emotion_test, y_emotion_pred))
    print(classification_report(y_emotion_test, y_emotion_pred))

    print("Sentiments (default parameters)")
    print(confusion_matrix(y_sentiment_test, y_sentiment_pred))
    print(classification_report(y_sentiment_test, y_sentiment_pred))

# 2.3.3
def base_mlp():
    print('==================================================')
    print("Base-MLP")
    encoded = vec.fit_transform(element[0] for element in json_load)
    emotions = [emotion[1] for emotion in json_load]
    sentiments = [sentiment[2] for sentiment in json_load]

    x_emotion_training, x_emotion_test, y_emotion_training, y_emotion_test = train_test_split(encoded, emotions, test_size=0.20, random_state=77)
    x_sentiment_training, x_sentiment_test, y_sentiment_training, y_sentiment_test = train_test_split(encoded, sentiments, test_size=0.20, random_state=77)

    classifier = MLPClassifier()
    classifier.fit(x_emotion_training, y_emotion_training)
    y_emotion_pred = classifier.predict(x_emotion_test)

    classifier.fit(x_sentiment_training, y_sentiment_training)
    y_sentiment_pred = classifier.predict(x_sentiment_test)

    print(confusion_matrix(y_emotion_test, y_emotion_pred))
    print(classification_report(y_emotion_test, y_emotion_pred))

    print(confusion_matrix(y_sentiment_test, y_sentiment_pred))
    print(classification_report(y_sentiment_test, y_sentiment_pred))

# 2.3.5
def top_dt():
    print('==================================================')
    print("Top-DT")
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [50, 100],
        'min_samples_split': [2, 5, 8]
    }

    encoded = vec.fit_transform(element[0] for element in json_load)
    emotions = [emotion[1] for emotion in json_load]
    sentiments = [sentiment[2] for sentiment in json_load]

    x_emotion_training, x_emotion_test, y_emotion_training, y_emotion_test = train_test_split(encoded, emotions, test_size=0.20, random_state=77)
    x_sentiment_training, x_sentiment_test, y_sentiment_training, y_sentiment_test = train_test_split(encoded, sentiments, test_size=0.20, random_state=77)

    dtc = tree.DecisionTreeClassifier()

    gs = GridSearchCV(estimator=dtc, param_grid=params)

    gs.fit(x_emotion_training, y_emotion_training)

    best_params = gs.best_params_ #{'criterion': 'gini', 'max_depth': 50, 'min_samples_split': 8}

    improved_dtc = tree.DecisionTreeClassifier(criterion=best_params['criterion'], max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
    improved_dtc.fit(x_emotion_training, y_emotion_training)
    improved_dtc_emotion_pred = improved_dtc.predict(x_emotion_test)
    
    print(confusion_matrix(y_emotion_test, improved_dtc_emotion_pred))
    print(classification_report(y_emotion_test, improved_dtc_emotion_pred))

    gs.fit(x_sentiment_training, y_sentiment_training)
    best_params == gs.best_params_

    improved_dtc = tree.DecisionTreeClassifier(criterion=best_params['criterion'], max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
    improved_dtc.fit(x_sentiment_training, y_sentiment_training)
    improved_dtc_sentiment_pred = improved_dtc.predict(x_sentiment_test)

    print(confusion_matrix(y_sentiment_test, improved_dtc_sentiment_pred))
    print(classification_report(y_sentiment_test, improved_dtc_sentiment_pred))

    #print('Improved emotions DTC score: {}'.format(improved_dtc_emotions_score)) #0.4148527528809219
    # print('Emotions score (entropy | max_depth: 50 | min_samples_split): {}'.format(emotion_score)) # 0.39593760912582937

# best_params = {'criterion': 'gini', 'max_depth': 50, 'min_samples_split': 8}
# print(best_params['min_samples_split'])
# top_dt()
# base_dt()
base_mlp()