import string
 
test_str = 'Taking a blood thinner such as aspirin would keep it from clotting!'
 
test_str = test_str.translate(str.maketrans('', '', string.punctuation))
print(test_str)