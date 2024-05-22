import torch


with open("./AliceInWonderLand.txt","r", encoding = "utf-8") as file:
    alice_text = file.read()
    charset = sorted(set(alice_text))



# TODO Add exception and edge case handling for the dictionaries 

#Simple assignment of an increasing index to each unique character in the dataset
strtoi = { ch : i for i, ch in enumerate(charset) }
# A reverse dictionary of strtoi. I'm too lazy to figure out if we can reverse a dictionary in python
itostr = { i : ch for i, ch in enumerate(charset) }

# Functions to form a tuple of vector of numbers and an string from a vector
encode = lambda s : [strtoi[i] for i in s]
decode = lambda s : "".join([itostr[i] for i in s])

# This works without having to crate a new secondary reversed lookup dictionary. But it's too cumbersome and complex to read
# decode = lambda s: "".join([list(strtoi.keys())[list(strtoi.values()).index(i)] for i in s])

print(decode(encode("hello")))