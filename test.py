import numpy as np

dataset = np.array([
['sunny', 'hot', 'high', 'false', 'Don\'t Play'],
])  

X = dataset[:, 0:-2]
y = dataset[:, 4]

print(X)
print(y)