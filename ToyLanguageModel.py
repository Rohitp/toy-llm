import torch
import torch.nn as nn
import torch.nn.functional as func


#using a bigram model here
class ToyLanguageModel(nn.Module):
    def __init__(self, charset_size):
        super().__init__()

        # The embedding dimension is also roughly the charset size here. 
        # This is determined emperically 
        # Word2Vec: 100 to 300 
        # GloVe: 50, 100
        # So somewhere in between?


        self.embedding_table = nn.Embedding(charset_size, charset_size)
        pass
        pass


    # Writing our own forward pass here to understand how this works. There are more efficient versions of this implementation online
    # We also try and reduce loss here -ln(1/charset_size) -> 1/charset_size is the probability of the next characte being predicted correctly
    def forward_pass(self, index, targets = None):
        logits = self.embedding_table(index)


        if targets is None:
            loss = None
        else: 

            # https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

            B, T, C = logits.shape

            #reshaping to get batch size x number of classes
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)

            loss = func.cross_entropy(logits, targets)

        return logits, loss 
    

    def generate(self, index, max_tokens):

        for i in range(max_tokens):

            logits, loss = self.forward_pass(index)

            logits = logits[:, -1, :]

            probabilities = func.softmax(logits, dim = -1)

            next_index = torch.multinomial(probabilities, num_samples=1)

            index = torch.cat((index, next_index), dim=1)

        return index