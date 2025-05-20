import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
import numpy as np
import random
import math
from typing import Optional

########################################################################################
# Encoder
########################################################################################

# RNN

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_latent, n_layers, dropout=0.0):
        super().__init__()
        
        self.enc_dim = hid_dim
        self.n_latent = n_latent
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.bn = nn.BatchNorm1d(n_latent)
        self.rnn = nn.GRU(input_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)
        self.fc = nn.Linear(hid_dim, n_latent)
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout
        
    def forward(self, src):
        self.rnn.flatten_parameters()
        #src = [batch size, src len, channel_dim]
        
        embedded = self.dropout(src)
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded)
        
        #outputs = [batch size, src len, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        latent = self.bn(self.fc(hidden[-1]))
        #outputs are always from the top hidden layer
        
        return latent

class GroupRNNEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_branch, n_latent, n_layers=1, dropout=0.0, device=None):
        super().__init__()

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.hid_dim = hid_dim
        self.n_latent = n_latent
        self.input_dim = input_dim
        self.n_branch = n_branch
        self.n_layers = n_layers
        self.group_rnn = nn.Sequential()
        for i in range(self.n_branch):
            self.group_rnn.add_module("rnn"+str(i+1), nn.GRU(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True).to(self.device))

        self.fc = nn.Linear(hid_dim*n_branch, n_latent)
        self.bn = nn.BatchNorm1d(n_latent)
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

    def forward(self, src):
        for i in range(self.n_branch):
            self.group_rnn[i].flatten_parameters()
        
        # interval = src.shape[1] // self.n_branch + 1
        group_hid = []
        for i in range(self.n_branch):
            group_hid.append(self.group_rnn[i](src)[-1][-1])

        group_hid = torch.cat(group_hid, dim=1)
        # group_hid = self.rnn(src)[-1][-1]
        hidden = self.bn(self.fc(group_hid))
        return hidden

########################################################################################

# Transformer

# Pytorch Version
class TransformerEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.hid_dim = configs.d_model
        # self.seq_len = configs.seq_len
        self.input_dim = configs.input_dim
        self.n_layers = configs.e_layers
        self.embedding = nn.Linear(self.input_dim, self.hid_dim)
        if configs.n_heads is None:
            self.nhead = self.hid_dim // 2
        else:
            self.nhead = configs.n_heads
        self.dropout = configs.dropout

        self.pos_enc = get_pos_encoder('fixed')(self.hid_dim, dropout=self.dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hid_dim, nhead=self.nhead, dim_feedforward=self.hid_dim*self.input_dim, dropout=self.dropout)
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        # self.pool_layer = nn.AdaptiveMaxPool1d(1)
        # self.linear = nn.Linear(self.hid_dim, self.input_dim)
        # self.bn = nn.BatchNorm1d(n_latent)

    def forward(self, src):
        B, L, N = src.shape
        src = src.permute(1, 0, 2)  # (time_window, batch_size, hid_dim)
        embedded = self.embedding(src)
        embedded = self.pos_enc(embedded)  # (time_window, batch_size, hid_dim)
        outputs = self.transformer_enc(embedded).permute(1, 0, 2) # (batch_size, time_window, hid_dim)
        # hidden = self.pool_layer(outputs).squeeze(2)
        # hidden = self.bn(self.linear(hidden))
        hidden = outputs.reshape(B,-1) # (batch_size, time_window*hid_dim)
        return hidden


# From https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


##################################################################################################################
# inverted Transformer
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding_inverted
class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(iTransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = False
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, 'fixed', 'h',
                                                    configs.dropout)
        self.class_strategy = None
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 5, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        B, L, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # B N E -> B N*E
        enc_out = enc_out.reshape(B,-1) 
        # B N E -> B N S -> B S N 
        # dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates 
        # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return enc_out


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        enc_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # return enc_out[:, -self.pred_len:, :]  # [B, L, D]
        return enc_out  # [B, N*E]


########################################################################################

# TemporalConvolutionNet

# From https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/master/networks/causal_cnn.py
class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.
    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        x_sl = x[:, :, :-self.chomp_size]
        return x_sl


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).
    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()
        dropout1 = nn.Dropout(dropout)
        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()
        dropout2 = nn.Dropout(dropout)
        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, dropout1, conv2, chomp2, relu2, dropout2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).
    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size, dropout):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size, dropout
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.
    Takes as input a three-dimensional tensor (`B`, `L_in`, `C`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a two-dimensional tensor (`B`, 'L_out', `C_reduced`).
    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param n_layers Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param n_latent Dimension of latent representation.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, reduced_size, input_len, 
                 output_len, n_layers=1, kernel_size=3, dropout=0.0):
        self.hid_dim = channels
        self.output_len = output_len
        self.input_dim = in_channels
        self.n_layers = n_layers
        self.dropout_rate = dropout
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, n_layers, reduced_size, kernel_size, dropout
        )
        # reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        # squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(input_len, output_len)
        self.network = torch.nn.Sequential(
            causal_cnn, linear
        )

    def forward(self, x):
        return self.network(x.transpose(1,2)).transpose(1,2)



########################################################################################
# Decoder
########################################################################################

class RNNDecoder(nn.Module):
    def __init__(self, output_dim, hid_dim, dropout=0.0, device=None):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        # self.n_layers = n_layers
        # self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRUCell(1, hid_dim, device=device)
        # self.rnn = BNGRUCell(1, hid_dim, device=device)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        # self.rnn.flatten_parameters()
        #input = [batch size, 1]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        # print(input.shape)
        # input = input.unsqueeze(2)
        # print(input.shape)
        #input = [batch size, 1, 1]

        embedded = self.dropout(input)
        #embedded = [batch size, 1, 1]
                
        hidden = self.rnn(embedded, hidden)
        # print(hidden.shape)
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [batch size, 1, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out( hidden )
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden

########################################################################################
# Seq2Seq model
########################################################################################
# class MultiSeq2Seq(nn.Module):
#     def __init__(self, encoder, decoder, device, hidden_regularizer=None, random_state=None):
#         super().__init__()
        
#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device
#         self.random_state = random_state
#         self.regularizer = hidden_regularizer
#         self.loss_reg = 0
        
#         # assert encoder.n_latent == decoder.hid_dim, \
#         #     "ouput dimension of encoder and input dimension of decoder must be equal!"

#     def build_fc_obs(self, time_window, aug_dim=0):
#         if aug_dim == 0:
#             self.fc_obs_x = nn.Linear(self.encoder.n_latent, self.encoder.input_dim*time_window)
#         if aug_dim > 0:
#             self.fc_obs_x = nn.Linear(aug_dim, self.encoder.input_dim*time_window)

#     def build_cus_enc(self, time_window):
#         self.cus_enc = nn.Sequential()
#         self.cus_enc.add_module("linear", nn.Linear(time_window, self.decoder.hid_dim - self.encoder.n_latent))
#         self.cus_enc.add_module("batch_norm", nn.BatchNorm1d(self.decoder.hid_dim - self.encoder.n_latent))
  
#     def forward(self, src, trg_len, hid_comp=None, random_seed=0):
#         self.loss_reg = torch.tensor(0.0).to(self.device)
#         random.seed(self.random_state + random_seed)
#         np.random.seed(self.random_state + random_seed)
#         #src = [batch size, src len, input_dim]
#         #trg = [batch size, trg len, output_dim]
        
#         batch_size = src.shape[0]
#         trg_channel_dim = self.decoder.output_dim
        
#         #tensor to store decoder outputs
#         outputs_y = torch.zeros(batch_size, trg_len, trg_channel_dim).to(self.device)
        
#         #last hidden state of the encoder is used as the initial hidden state of the decoder
#         if hid_comp is None:
#             hidden = self.encoder(src)
#             # hidden = hidden[-1:]
#             hidden = hidden.reshape(batch_size,-1)
#             outputs_x = self.fc_obs_x(hidden)
#         # print(hidden.size())
#         else:
#             cus = src[:,:,0]
#             cus_trans = self.cus_enc(cus)
#             l = np.random.choice(len(hid_comp), 1)[0]
#             hidden = torch.cat([cus_trans, hid_comp[l]], dim=1)
#             outputs_x = self.fc_obs_x(cus_trans)

#         #first input to the decoder is generated from fc_obs 
#         input = torch.zeros(batch_size, 1).to(self.device) 

#         if self.regularizer is not None:
#             self.loss_reg += self.regularizer(hidden, self.device, self.random_state)

#         outputs_x = outputs_x.contiguous().view(batch_size, -1, self.encoder.input_dim)

#         for t in range(0, trg_len):
            
#             output, hidden= self.decoder(input, hidden)
#             # hidden = self.encoder.bn(hidden[0]).reshape(1,batch_size,-1)
#             outputs_y[:,t] = output
#             if self.regularizer is not None:
#                 self.loss_reg += self.regularizer(hidden, self.device, self.random_state)

        
#         return outputs_x, outputs_y

class MultiSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, hidden_regularizer=None, random_state=None):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.random_state = random_state
        self.regularizer = hidden_regularizer
        self.loss_reg = 0
        
        # assert encoder.n_latent == decoder.hid_dim, \
        #     "ouput dimension of encoder and input dimension of decoder must be equal!"

    def build_fc_obs(self, time_window):
        self.fc_obs_x = nn.Linear(self.decoder.hid_dim, (1+self.encoder.input_dim)*time_window)


    def build_cus_enc(self, time_window):
        self.cus_enc = nn.Sequential()
        self.cus_enc.add_module("linear", nn.Linear(time_window, self.decoder.hid_dim - self.encoder.n_latent))
        # self.cus_enc.add_module("batch_norm", nn.BatchNorm1d(self.decoder.hid_dim - self.encoder.n_latent))

    def embed(self, src, mode='reduced'):
        Mode = {'reduced': 0, 'full': 1}
        with torch.no_grad():
            if Mode[mode]==0:
                hidden = self.encoder(src[:,:,1:]).reshape(src.shape[0], -1)
                return hidden
            else:
                hidden = self.encoder(src[:,:,1:]).reshape(src.shape[0], -1)
                cus_trans = self.cus_enc(src[:,:,0])
                return torch.cat([cus_trans, hidden], dim=1)
            
    def forward(self, src, trg_len, random_seed=0):
        self.loss_reg = torch.tensor(0.0).to(self.device)
        #src = [batch size, src len, input_dim]
        #trg = [batch size, trg len, output_dim]
        
        batch_size = src.shape[0]
        trg_channel_dim = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs_y = torch.zeros(batch_size, trg_len, trg_channel_dim).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder

        hidden = self.encoder(src[:,:,1:])
        # hidden = hidden[-1:]
        hidden = hidden.reshape(batch_size,-1)

        # print(hidden.size())
        cus = src[:,:,0]
        cus_trans = self.cus_enc(cus)
        self.loss_reg += torch.sum(torch.mean(torch.abs(cus_trans), dim=0))

        hidden = torch.cat([cus_trans, hidden], dim=1)
        outputs_x = self.fc_obs_x(hidden)

        #first input to the decoder is generated from fc_obs 
        input = torch.zeros(batch_size, 1).to(self.device) 

        if self.regularizer is not None:
            self.loss_reg += self.regularizer(hidden, self.device, self.random_state)

        outputs_x = outputs_x.contiguous().view(batch_size, -1, self.encoder.input_dim+1)

        for t in range(0, trg_len):
            
            output, hidden= self.decoder(input, hidden)
            # hidden = self.encoder.bn(hidden[0]).reshape(1,batch_size,-1)
            outputs_y[:,t] = output
            if self.regularizer is not None:
                self.loss_reg += self.regularizer(hidden, self.device, self.random_state)

        
        return outputs_x, outputs_y
    
########################################################################################
# The abstract network architecture of embedding representation learning
########################################################################################
# base class
class RepresentLearn(nn.Module):

    def __init__(self, encoder, time_window, input_dim, enc_dim, out_dim, random_state=None, device=torch.device('cpu')):
        super(RepresentLearn, self).__init__()

        self.encoder = encoder.to(device)
        self.time_window = time_window
        self.input_dim = input_dim
        self.enc_dim = enc_dim
        self.out_dim = out_dim
        self.random_state = random_state
        self.device = device

    def embed(self, src):

        with torch.no_grad():
            hidden = self.encoder(src).reshape(src.shape[0], -1)
            return hidden
            
    def forward(self, X):
        raise AttributeError("This class does not contain forward() method.")



class Seq2SeqRepresent(RepresentLearn):
    def __init__(self, decoder, hidden_regularizer=None,
        *args,
        **kwargs):
        super().__init__(*args, **kwargs)

        self.decoder = decoder.to(self.device)
        self.regularizer = hidden_regularizer
        self.loss_reg = 0
        
    def forward(self, src, trg_len):
        self.loss_reg = torch.tensor(0.0).to(self.device) 
        random.seed(self.random_state)
        
        batch_size = src.shape[0]
        trg_channel_dim = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_channel_dim).to(self.device)        
        input = torch.zeros(batch_size, 1).to(self.device)        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(src)
        hidden = hidden.reshape(batch_size,-1)

        # if self.regularizer is not None:
        #     self.loss_reg += self.regularizer(hidden, self.device, self.random_state)
        if self.regularizer == 'l1':
            self.loss_reg += torch.mean(torch.abs(hidden), axis=0).sum()    

        for t in range(0, trg_len):

            output, hidden = self.decoder(input, hidden)
            outputs[:,t,:] = output
            
            # if self.regularizer is not None:
            #     self.loss_reg += self.regularizer(hidden, self.device, self.random_state)
            
        return outputs


class TransformerRepresent(RepresentLearn):

    def __init__(self,         
        *args, output_len, target_dim=1, hidden_regularizer=None,
        **kwargs):
        super().__init__(*args, **kwargs)
        # self.encoder = CausalCNNEncoder(input_dim, enc_dim, hid_num, n_layers, dropout=dropout)
        ########################################################################################
        # override        
        ########################################################################################

        self.output_len = output_len
        self.projector = nn.Linear(self.time_window*self.enc_dim, self.out_dim, bias=True).to(self.device)
        self.predictor = nn.Linear(self.out_dim, self.output_len*target_dim, bias=True).to(self.device)
        self.regularizer = hidden_regularizer
        self.loss_reg = 0

    # override
    def embed(self, src):

        with torch.no_grad():            
            state = self.encoder(src).reshape(src.shape[0], -1)
            hidden = self.projector(state)                
                
            return hidden

    def forward(self, X, noise_sig=None):
        self.loss_reg = torch.tensor(0.0).to(self.device)
        batch_size = X.shape[0]
        state = self.encoder(X).reshape(batch_size,-1)
        hidden = self.projector(state)
        preds = self.predictor(hidden).reshape(batch_size,self.output_len,-1)
        if self.regularizer == 'l1':
            # self.loss_reg += self.regularizer(hidden, self.device, self.random_state)
            self.loss_reg += torch.mean(torch.abs(hidden), axis=0).sum()
        # hidden = self.trans_layer(state.reshape(batch_size, -1))
        if noise_sig is None:
            return preds, None
        else:
            K = len(noise_sig)
            # noi = -1*noise_sig[0] + 2 * noise_sig[0] * torch.rand((K,batch_size,self.out_dim)).to(self.device)
            # noi = noise_sig[0] * torch.randn((K,batch_size,self.out_dim)).to(self.device)
            noi = torch.tensor(np.random.choice([-1,1], (K,batch_size,self.out_dim))).to(self.device) * noise_sig[0]
            preds_noi = self.predictor( hidden + noi ).reshape(K,batch_size,self.output_len,-1)
            # preds_noi = self.predictor(self.projector(self.encoder(X+torch.rand(X.size())*noise_sig)))
            return preds, preds_noi    


class NoiseAmpRepresent(RepresentLearn):

    def __init__(self,         
        *args, output_len, target_dim=1, hidden_regularizer=None,
        **kwargs):
        super().__init__(*args, **kwargs)
        # self.encoder = CausalCNNEncoder(input_dim, enc_dim, hid_num, n_layers, dropout=dropout)
        ########################################################################################
        # override        
        ########################################################################################

        self.output_len = output_len
        self.projector = nn.Linear(self.input_dim*self.enc_dim, self.out_dim, bias=True).to(self.device)
        self.predictor = nn.Linear(self.out_dim, self.output_len*target_dim, bias=True).to(self.device)
        self.regularizer = hidden_regularizer
        self.loss_reg = 0

    # override
    def embed(self, src):

        with torch.no_grad():            
            state = self.encoder(src).reshape(src.shape[0], -1)
            hidden = self.projector(state)                
                
            return hidden

    def forward(self, X, noise_sig=None):
        self.loss_reg = torch.tensor(0.0).to(self.device)
        batch_size = X.shape[0]
        state = self.encoder(X).reshape(batch_size,-1)
        hidden = self.projector(state)
        preds = self.predictor(hidden).reshape(batch_size,self.output_len,-1)
        if self.regularizer == 'l1':
            # self.loss_reg += self.regularizer(hidden, self.device, self.random_state)
            self.loss_reg += torch.mean(torch.abs(hidden), axis=0).sum()
        # hidden = self.trans_layer(state.reshape(batch_size, -1))
        if noise_sig is None:
            return preds, None
        else:
            K = len(noise_sig)
            # noi = -1*noise_sig[0] + 2 * noise_sig[0] * torch.rand((K,batch_size,self.out_dim)).to(self.device)
            # noi = noise_sig[0] * torch.randn((K,batch_size,self.out_dim)).to(self.device)
            noi = torch.tensor(np.random.choice([-1,1], (K,batch_size,self.out_dim))).to(self.device) * noise_sig[0]
            preds_noi = self.predictor( hidden + noi ).reshape(K,batch_size,self.output_len,-1)
            # preds_noi = self.predictor(self.projector(self.encoder(X+torch.rand(X.size())*noise_sig)))
            return preds, preds_noi    

