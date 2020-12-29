"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers , bert_layers 
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
    
    
class BiDAF_charCNN(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF_charCNN, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        
        self.char_emb = layers.CharEmbedding(char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        
        self.hwy = layers.HighwayEncoder(2, 2*hidden_size)

        self.enc = layers.RNNEncoder(input_size=2*hidden_size,
                                     hidden_size=2*hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * 2*hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * 2*hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb_w = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb_w = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
        
        c_emb_cc = self.char_emb(cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb_cc = self.char_emb(qc_idxs)         # (batch_size, q_len, hidden_size)
        
        c_emb = self.hwy(torch.cat([c_emb_w,c_emb_cc],axis=-1))
        q_emb = self.hwy(torch.cat([q_emb_w,q_emb_cc],axis=-1))

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAF_charCNN_BERTEnc(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.,twist_embeddings=True):
        super(BiDAF_charCNN_BERTEnc, self).__init__()
        
        ###
        self.twist_embeddings = twist_embeddings
        idx_list = []
        for i in range(hidden_size):
            idx_list.append(i)
            idx_list.append(hidden_size+i)
        self.register_buffer('idx_twist',torch.tensor(idx_list))
        ###
        
        
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        
        self.char_emb = layers.CharEmbedding(char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        
        self.hwy = layers.HighwayEncoder(2, 2*hidden_size)

        self.enc = bert_layers.BertEncoder(n_layers=6, #n_layers=3,
                                           d_feature=2*hidden_size, 
                                           n_heads=8,
                                           out_size=2*hidden_size,
                                           d_ff=2048,
                                           #d_ff = 2*hidden_size, 
                                           dropout_prob=0.1,
                                           #dropout_prob=drop_prob,
                                           ff_activation=F.relu)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)
    
    def twist(self,a,b):
        assert a.shape  == b.shape  , 'tensors to be twisted need to have the same size'
        idx = self.idx_twist.repeat(a.shape[0],a.shape[1],1)
        c = torch.cat([a,b],axis=-1)
        return torch.gather(c,-1,idx)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb_w = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb_w = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
        
        c_emb_cc = self.char_emb(cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb_cc = self.char_emb(qc_idxs)         # (batch_size, q_len, hidden_size)
        
        if self.twist_embeddings:
            c_emb = self.hwy(self.twist(c_emb_w,c_emb_cc))
            q_emb = self.hwy(self.twist(q_emb_w,q_emb_cc)) 
        else:
            c_emb = self.hwy(torch.cat([c_emb_w,c_emb_cc],axis=-1))
            q_emb = self.hwy(torch.cat([q_emb_w,q_emb_cc],axis=-1))

        c_enc = self.enc(c_emb)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAF_charCNN_BERTEnc_BERTMod(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.,twist_embeddings=False):
        super(BiDAF_charCNN_BERTEnc_BERTMod, self).__init__()
        
        ###
        self.twist_embeddings = twist_embeddings
        idx_list = []
        for i in range(hidden_size):
            idx_list.append(i)
            idx_list.append(hidden_size+i)
        self.register_buffer('idx_twist',torch.tensor(idx_list))
        ###
        
        
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        
        self.char_emb = layers.CharEmbedding(char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        
        self.hwy = layers.HighwayEncoder(2, 2*hidden_size)

        self.enc = bert_layers.BertEncoder(n_layers=4, #n_layers=4,
                                           d_feature=2*hidden_size, 
                                           n_heads=8,
                                           out_size=2*hidden_size,
                                           d_ff=2048,
                                           #d_ff = 2*hidden_size, 
                                           dropout_prob=0.1,
                                           #dropout_prob=drop_prob,
                                           ff_activation=F.relu)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)
        
        self.mod = bert_layers.BertEncoder(n_layers=6, #n_layers=3,
                                           d_feature=8*hidden_size, 
                                           n_heads=8,
                                           out_size=2*hidden_size,
                                           d_ff=2048,
                                           #d_ff = 2*hidden_size, 
                                           dropout_prob=0.1,
                                           #dropout_prob=drop_prob,
                                           ff_activation=F.relu)

        # self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
        #                              hidden_size=hidden_size,
        #                              num_layers=2,
        #                              drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)
    
    def twist(self,a,b):
        assert a.shape  == b.shape  , 'tensors to be twisted need to have the same size'
        idx = self.idx_twist.repeat(a.shape[0],a.shape[1],1)
        c = torch.cat([a,b],axis=-1)
        return torch.gather(c,-1,idx)
        

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb_w = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb_w = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
        
        c_emb_cc = self.char_emb(cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb_cc = self.char_emb(qc_idxs)         # (batch_size, q_len, hidden_size)
        
        if self.twist_embeddings:
            c_emb = self.hwy(self.twist(c_emb_w,c_emb_cc))
            q_emb = self.hwy(self.twist(q_emb_w,q_emb_cc)) 
        else:
            c_emb = self.hwy(torch.cat([c_emb_w,c_emb_cc],axis=-1))
            q_emb = self.hwy(torch.cat([q_emb_w,q_emb_cc],axis=-1))

        c_enc = self.enc(c_emb)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
