# Code by Sarah Wiegreffe (saw@gatech.edu)
# Fall 2019

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class ClassificationTransformer(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, word_to_ix, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        '''
        :param word_to_ix: dictionary mapping words to unique indices
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        '''        
        super(ClassificationTransformer, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.vocab_size = len(word_to_ix)
        
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        self.embedding_layer = nn.Embedding(self.vocab_size, self.word_embedding_dim)
        self.posit_embed_layer = nn.Embedding(self.max_length, self.word_embedding_dim)
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        
        self.w1 = nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.w2 = nn.Linear(self.dim_feedforward, self.hidden_dim)

        
        self.fl = nn.Linear(self.hidden_dim, 1)
        self.final_softmax = nn.Softmax(dim=1)
        self.final_sigmoid = nn.Sigmoid()

        
    def forward(self, inputs):
        '''
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups. 

        :returns: the model outputs. Should be normalized scores of shape (N,1).
        '''
        return self.final_layer(self.feedforward_layer(self.multi_head_attention(self.embed(inputs))))
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        N, T = [*inputs.size()]
        sample_output = self.embedding_layer(inputs[0])
        H = sample_output.size(1)
        embeds = torch.empty(N, T, H)
        for i in range(N):
            embeds[i] = self.embedding_layer(inputs[i]) + self.posit_embed_layer(torch.arange(0, 43))
        return embeds

        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        N, T, H = [*inputs.size()]

        Q1 = self.q1(inputs)
        K1 = self.k1(inputs)
        V1 = self.v1(inputs)
        attn1 = torch.bmm(self.softmax(torch.bmm(Q1, K1.permute(0, 2, 1)) / (K1.size(2) ** 0.5)), V1)
        # print(attn1.size())

        Q2 = self.q2(inputs)
        K2 = self.k2(inputs)
        V2 = self.v2(inputs)
        attn2 = torch.bmm(self.softmax(torch.bmm(Q2, K2.permute(0, 2, 1)) / (K2.size(2) ** 0.5)), V2)
        # print(attn2.size())

        return self.norm_mh(inputs + self.attention_head_projection(torch.cat((attn1, attn2), dim=2)))
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """

        return self.norm_mh(inputs + self.w2(self.w1(inputs).clamp(min=0)))
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,1)
        """
        
        ft = inputs[:, 0, :]

        return self.final_sigmoid(self.fl(ft))
        

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True