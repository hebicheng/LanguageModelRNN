import torch
import torch.nn as nn


class RNNModel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.5):
        ''' 该模型包含以下几层:
            - 词嵌入层
            - 一个循环神经网络层 LSTM
            - 一个线性层，从hidden state到输出单词表
            - 一个dropout层，用来做regularization
        '''
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        ''' Forward pass:
            - word embedding
            - 输入循环神经网络
            - 一个线性层从hidden state转化为输出单词表
        '''
        emb = self.drop(self.encoder(input))
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())
        return (weight.new_zeros((1, bsz, self.hidden_size), requires_grad=requires_grad),
                weight.new_zeros((1, bsz, self.hidden_size), requires_grad=requires_grad))
        