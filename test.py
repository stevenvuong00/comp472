import numpy as np
import string

# dataset = np.array([
# ['sunny', 'hot', 'high', 'false', 'Don\'t Play'],
# ])  

# X = dataset[:, 0:-2]
# y = dataset[:, 4]

# print(X)
# print(y)

arr1 = np.array([[1, 2], [2], [3]])
arr2 = np.array([[3], [2] ,[1]])

sizeArr1 = arr1.size
maxSublist = max(arr1, key=len) # returns the longest sub list

print(sizeArr1)
print(maxSublist)

# padding the arrays
a = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
b = np.zeros([len(a),len(max(a, key = lambda x: len(x)))])
for i, j in enumerate(a):
    b[i][0:len(j)] = j

# print(a)
# print(b)

# print ("1st array : ", arr1)  
# print ("2nd array : ", arr2)  

# out_arr = np.add(arr1, arr2)  
# print ("added array : ", out_arr)  

# make all the list remove the last 2 elements first
# pad with "" until the end
# then readd the emotion and sentiment
# now we have ndarray of m x m 

# https://stackoverflow.com/questions/47666539/numpy-zeros-and-string-data-type blank but in ''

# simulate json_load
arr3 = [['Taking a blood thinner such as aspirin would keep it from clotting!', 'approval', 'positive'], ['I am glad that the message touched you, I really am.', 'gratitude', 'positive']]
# np_arr3 = np.array(arr3)

# get all the last 2 elements of the data set
targets_emotion = [element[1] for element in arr3]
print(targets_emotion)

targets_sentiment = [element[2] for element in arr3]
print(targets_sentiment)

# list of only the words without label
only_words = [element[0] for element in arr3]
print(only_words)

# filter punctuation
clean_words = []
for element in only_words:
    clean = element.translate(str.maketrans('', '', string.punctuation))
    clean_words.append(clean)

print(clean_words) 

# transform each sublist into a list of list
words_list = [element.split() for element in only_words]
print(words_list)

# get max length of a sublist, that's the length number of features (words)
maxLenSublist = max(words_list, key=len)
print("max sublength: ")
print(len(maxLenSublist))

# pad the lists
# dtype='<U25' accept all strings length 25 --> creating fresh 2d np array with the needed length
# could change U25
padded_words = np.zeros((len(arr3), len(maxLenSublist)), dtype='<U25')

# adding all the words to the new array
for i in range(len(arr3)):
    for j in range(len(words_list[i])):
        # need to prevent index uot of bound, depends on the length of the orinigal word list
        padded_words[i][j] = words_list[i][j]
print(padded_words)

# re-adding the emotion and sentiments
training_set = padded_words.copy().tolist()
for i in range(len(training_set)):
    training_set[i].append(targets_emotion[i])
    training_set[i].append(targets_sentiment[i])



# print(targets_emotion[0])
print(training_set)
