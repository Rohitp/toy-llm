import torch.nn as nn


#using a bigram model here
class ToyLanguageModel(nn.Module):
    def __init__(self, charset_size):
        super().__init__()
        self.embedding_table = nn.Embedding(charset_size, charset_size)


    # Writing our own forward pass here to understand how this works. There are more efficient versions of this implementation online
    # We also try and reduce loss here -ln(1/charset_size) -> 1/charset_size is the probability of the next characte being predicted correctly
    def forward_pass(self, index, targets):
        logits = self.embedding_table(index)