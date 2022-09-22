import json
import gzip
# import matplotlib.pyplot as plt

f = gzip.open('C:\\Users\\15146\\School\\COMP 472 Labs\\project 1\\comp472\\goemotions.json.gz', 'rb')

json_load = json.load(f)
# jsonArr = np.array(json_load)
# x = [jsonArr[key] for key in jsonArr]
# print(x)
print(json.dumps(json_load, indent=4))

