import torch


# Device check for runpod vs local.
DEVICE = "Cuda" if torch.cuda.is_available() else "CPU"

# Block size of the token. TODO Iterate on the ideal block size. This can be at a word level as well, but this is getting too nig
BLOCK_SIZE = 8

#Size of the batch for parelel cuda processing
BATCH_SIZE = 4

with open("./AliceInWonderLand.txt","r", encoding = "utf-8") as file:
    alice_text = file.read()

charset = sorted(set(alice_text))
charset_size = len(charset)



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


# Forming one long vector with the datalll
data = torch.tensor(encode(alice_text), dtype = torch.long)


# forming a test train split with the data

train_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]



# Batching data into chunks with source and target data mapping it 
def batch_for_cuda():
    x = torch.randint(high=len(train_data) - BLOCK_SIZE, size= (BATCH_SIZE,))
    offset = 0
    stacks_source = []
    stacks_target = []
    for i in range(BATCH_SIZE):
        x = train_data[i+offset:BLOCK_SIZE+offset+i]
        y = train_data[i+offset+1:BLOCK_SIZE+offset+1+i]

        """I'm sure there is a better way to do this, either with more intellignet use of torch functions, cat vs stack
        Or with a more pythonic one liner. For now too lazy to figure that out
        TODO: Refactor to a cleaner approach"""

        stacks_source.append(train_data[i+offset:BLOCK_SIZE+offset+i])
        stacks_target.append(train_data[i+offset+1:BLOCK_SIZE+offset+1+i])
        offset += BLOCK_SIZE

    source = torch.stack(stacks_source)
    target = torch.stack(stacks_target)
    return source, target

batch_for_cuda()
# print(batch_for_cuda())


# Batches and gets the next predicted character for any sequence of characters
for i in range(1, BLOCK_SIZE):
    current = [train_data[:i]]
    next = [train_data[i]]
    # print( "For " + str(current) + "Next is" + str(next))



# print(len(test_data))