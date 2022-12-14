import json
import gzip
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

f = gzip.open('goemotions.json.gz', 'rb')

json_load = json.load(f)

emotionsList =  [list[1] for list in json_load]
sentiments_list =  [list[2] for list in json_load]

def plot():
    plt.hist(emotionsList, bins = 55)
    plt.subplots_adjust(bottom = 0.3)
    plt.xticks(rotation = -90, size=5)
    plt.yticks(size=7)
    plt.title("Histogram of the Distribution of Emotions")
    plt.xlabel("Emotions")
    plt.ylabel("Number of Occurences")
    plt.savefig("outputs/graphs/Emotions_Histogram.pdf")
    plt.show()

    plt.hist(sentiments_list)
    plt.subplots_adjust(bottom = 0.3)
    plt.xticks(size=7)
    plt.yticks(size=7)
    plt.title("Histogram of the Distribution of Emotions")
    plt.xlabel("Sentiments")
    plt.ylabel("Number of Occurences")
    plt.savefig("outputs/graphs/Sentiments_Histogram.pdf")
    plt.show()

plot()