import json
import gzip
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

f = gzip.open('goemotions.json.gz', 'rb')

json_load = json.load(f)

emotionsList =  [list[1] for list in json_load]
sentimentsList =  [list[2] for list in json_load]

# plt.hist(emotionsList, bins = 55)
# plt.subplots_adjust(bottom = 0.3)
# plt.xticks(rotation = -90, size=5)
# plt.yticks(size=7)
# plt.title("Histogram of the Distribution of Emotions")
# plt.xlabel("Emotions")
# plt.ylabel("Number of Occurences")
# plt.savefig("Emotions Histogram")
# plt.show()

# plt.hist(sentimentsList)
# plt.subplots_adjust(bottom = 0.3)
# plt.xticks(size=7)
# plt.yticks(size=7)
# plt.title("Histogram of the Distribution of Emotions")
# plt.xlabel("Sentiments")
# plt.ylabel("Number of Occurences")
# plt.savefig("Sentiments Histogram")
# plt.show()

