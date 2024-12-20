import torch
import torch.nn.functional as func

# import ToyLanguageModel

from ToyLanguageModel import ToyLanguageModel


# Device check for runpod vs local.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Block size of the token. TODO Iterate on the ideal block size. This can be at a word level as well, but this is getting too nig
BLOCK_SIZE = 8

#Size of the batch for parelel cuda processing
BATCH_SIZE = 4

# Iterations for learning
ITERATIONS = 1000

# Generating a max batch of untrained tokens at once

MAX_TOKENS = 500

# Picked a number. Need to experiment here
# https://x.com/karpathy/status/801621764144971776?lang=en - though this might be a joke
# https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method
LEARNING_RATE = 3e-4



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


    # random indices to compare with
    rand_tensor = torch.randint(high=len(train_data) - BLOCK_SIZE, size= (BATCH_SIZE,))
    offset = 0
    stacks_source = []
    stacks_target = []
    for i in rand_tensor:

        x = train_data[i+offset:BLOCK_SIZE+offset+i]
        y = train_data[i+offset+1:BLOCK_SIZE+offset+1+i]

        """I'm sure there is a better way to do this, either with more intellignet use of torch functions, cat vs stack
        Or with a more pythonic one liner. For now too lazy to figure that out
        TODO: Refactor to a cleaner approach"""

        stacks_source.append(train_data[i+offset:BLOCK_SIZE+offset+i])
        stacks_target.append(train_data[i+offset+1:BLOCK_SIZE+offset+1+i])
        offset += BLOCK_SIZE

    source = torch.stack(stacks_source).to(DEVICE)
    target = torch.stack(stacks_target).to(DEVICE)

    return source, target




toyLM = ToyLanguageModel(charset_size)
m = toyLM.to(DEVICE)

context = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
generated = decode(m.generate(context, max_tokens=MAX_TOKENS)[0].tolist())





# Picking AdamW because of weight decay 
# https://towardsdatascience.com/why-adamw-matters-736223f31b5d
# model.parameters() is is charset_size x charset_size array of untrained values
optimiser = torch.optim.AdamW(toyLM.parameters(), lr=LEARNING_RATE)

for i in range(ITERATIONS):
    sources, targets = batch_for_cuda()
    logits, loss = toyLM.forward_pass(sources, targets)
    



