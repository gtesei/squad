import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# def create_tensor(t,dtype=None):
#     """Create tensor from list of lists"""
#     if not dtype:
#         return torch.tensor(t)
#     else:
#         return torch.tensor(t,dtype=dtype)


# def display_tensor(t, name):
#     """Display shape and tensor"""
#     print(f'{name} shape: {t.shape}\n')
#     print(f'{t}\n')
    
    
    
# q = create_tensor([[1, 0, 0], [0, 1, 0]])
# display_tensor(q, 'query')
# k = create_tensor([[1, 2, 3], [4, 5, 6]])
# display_tensor(k, 'key')
# v = create_tensor([[0, 1, 0], [1, 0, 1]],dtype=torch.float32)
# display_tensor(v, 'value')
# m = create_tensor([[0, 0], [-1e9, 0]])
# display_tensor(m, 'mask')

# q_with_batch = q[None,:]
# display_tensor(q_with_batch, 'query with batch dim')
# k_with_batch = k[None,:]
# display_tensor(k_with_batch, 'key with batch dim')
# v_with_batch = v[None,:]
# display_tensor(v_with_batch, 'value with batch dim')
# m_bool = create_tensor([[True, True], [False, True]],dtype=torch.bool)
# display_tensor(m_bool, 'boolean mask')


def dotProductAttention(query, key, value, mask):

    assert query.shape[-1] == key.shape[-1] == value.shape[-1], "Embedding dimensions of q, k, v aren't all the same"

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # Save depth/dimension of the query embedding for scaling down the dot product
    depth = query.shape[-1]

    # Calculate scaled query key dot product according to formula above
    dots =  torch.matmul(query, torch.transpose(key, -1, -2)) / math.sqrt(depth)
    
    # Apply the mask
    if mask is not None: # The 'None' in this line does not need to be replaced
        dots = torch.where(mask, dots, torch.full_like(dots, -1e9))
    
    # Softmax formula implementation
    # Use trax.fastmath.logsumexp of dots to avoid underflow by division by large numbers
    # Hint: Last axis should be used and keepdims should be True
    # Note: softmax = e^(dots - logsumexp(dots)) = E^dots / sumexp(dots)
    logsumexp = torch.logsumexp(dots,axis=-1,keepdims=True)

    # Take exponential of dots minus logsumexp to get softmax
    # Use jnp.exp()
    dots = torch.exp(dots-logsumexp)

    # Multiply dots by value to get self-attention
    # Use jnp.matmul()
    attention = torch.matmul(dots,value)

    ## END CODE HERE ###
    
    return attention


#dotProductAttention(q_with_batch, k_with_batch, v_with_batch, m_bool)


class BertSelfAttention(nn.Module):
    
    def __init__(self,d_feature,n_heads):
        super().__init__()
        
        assert d_feature % n_heads == 0
        d_head = d_feature // n_heads
        
        self.n_heads = n_heads
        self.d_feature = d_feature
        self.d_head = int(self.d_feature / self.n_heads)
        
        self.query = nn.Linear(self.d_feature , self.d_feature )
        self.key = nn.Linear(self.d_feature , self.d_feature)
        self.value = nn.Linear(self.d_feature , self.d_feature)
        
        self.out = nn.Linear(self.d_feature , self.d_feature)
        
    
    def compute_attention_heads(self, x):
        #new_x_shape = x.size()[:-1] + (self.n_heads, self.d_head)
        #x = x.view(*new_x_shape)
        #return x.permute(0, 2, 1, 3)
        batch_size = x.shape[0]
        seqlen = x.shape[1]
        x = x.view(batch_size, seqlen, self.n_heads, self.d_head)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size*self.n_heads,seqlen,self.d_head)
        return x
    
    def compute_attention_output(self, x):
        #x = x.permute(0, 2, 1, 3).contiguous()
        #new_context_layer_shape = x.size()[:-2] + (self.d_feature ,)
        #x = x.view(*new_context_layer_shape)
        #return x
        seqlen = x.shape[-2]
        x = x.view(-1,self.n_heads,seqlen,self.d_head)
        x = x.permute(0,2,1,3)
        return x.reshape(-1, seqlen, self.n_heads * self.d_head)
    
    def forward(self,hidden_states,attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.compute_attention_heads(mixed_query_layer)
        key_layer = self.compute_attention_heads(mixed_key_layer)
        value_layer = self.compute_attention_heads(mixed_value_layer)
        
        attention_scores = dotProductAttention(query_layer, key_layer, value_layer, attention_mask) ## bidirectional self-attention 
        
        attention_out = self.compute_attention_output(attention_scores)
        
        return self.out(attention_out)
    
# bsa = BertSelfAttention(d_feature=512, n_heads=8)
# x = torch.randn(64,20,512)
# xa = bsa(x)
# assert list(xa.shape) == [64, 20, 512]



class BertLayer(nn.Module):
    
    def __init__(self,d_feature,n_heads,d_ff,dropout_prob,ff_activation=F.relu):
        super().__init__()
        self.n_heads = n_heads
        self.d_feature = d_feature
        self.d_head = int(self.d_feature / self.n_heads)
        
        self.dropout_prob = dropout_prob
        self.ff_activation = ff_activation
        self.d_ff = d_ff
        
        self.attention = BertSelfAttention(d_feature,n_heads)
        self.layer_norm_1 = nn.LayerNorm(d_feature)
        self.layer_norm_2 = nn.LayerNorm(d_feature)
        self.dense1 = nn.Linear(self.d_feature , self.d_ff)
        self.dense2 = nn.Linear(self.d_ff , self.d_feature)
    
    
    def forward(self,input_tensor,attention_mask=None):
        
        ## multi-head attention layer 
        self_attention_outputs = self.attention(input_tensor,attention_mask)
        self_attention_outputs = F.dropout(self_attention_outputs, self.dropout_prob, self.training)
        self_attention_outputs = self.layer_norm_2(self_attention_outputs+input_tensor)
        
        ## fw layer 
        x = self.dense1(x)
        x = self.ff_activation(x)
        x = F.dropout(x, self.dropout_prob, self.training)
        x = self.dense2(x)
        x = F.dropout(x, self.dropout_prob, self.training)
        x = self.layer_norm_2(x+self_attention_outputs)
        
        return x 
        
# bsa = BertLayer(d_feature=512, n_heads=8,d_ff=2048,dropout_prob=0.1,ff_activation=F.relu)
# x = torch.randn(64,20,512)
# xa = bsa(x)
# assert list(xa.shape) == [64, 20, 512]
            
        
        
class BertEncoder(nn.Module):
    
    def __init__(self,n_layers,d_feature,n_heads,out_size,d_ff,dropout_prob,ff_activation=F.relu):
        super().__init__()
        self.dropout_prob = dropout_prob
        
        self.layer = nn.ModuleList([BertLayer(d_feature=d_feature, 
                                              n_heads=n_heads,
                                              d_ff=d_ff,
                                              dropout_prob=dropout_prob,
                                              ff_activation=ff_activation) for _ in range(n_layers)])
        self.dense = nn.Linear(d_feature , out_size)
    
    def forward(self,x,attention_mask=None):
        for i, layer_module in enumerate(self.layer):
            x = layer_module(x,attention_mask=attention_mask)
        x = self.dense(x)
        x = F.dropout(x, self.dropout_prob, self.training)
        return x 
        
    

# bsa = BertEncoder(n_layers=6,d_feature=512, n_heads=8,d_ff=2048,dropout_prob=0.1,ff_activation=F.relu)
# x = torch.randn(64,20,512)
# xa = bsa(x)
# assert list(xa.shape) == [64, 20, 512]
