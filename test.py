import numpy as np

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

# padding the 
a = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
b = np.zeros([len(a),len(max(a, key = lambda x: len(x)))])
for i, j in enumerate(a):
    b[i][0:len(j)] = j

print(a)
print(b)

# print ("1st array : ", arr1)  
# print ("2nd array : ", arr2)  

# out_arr = np.add(arr1, arr2)  
# print ("added array : ", out_arr)  

# make all the list remove the last 2 elements first
# pad with "" until the end
# then readd the emotion and sentiment
# now we have ndarray of m x m 