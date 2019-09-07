# -*- coding: utf-8 -*-

'''
Janurary 2018 by Wei Li
liweihfyz@sjtu.edu.cn
https://www.github.cim/leviswind/transformer-pytorch
'''

from modules import *
from torch.autograd import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, params, encoder_vocab, decoder_vocab):
        '''Attention is all you need. https://arxiv.org/abs/1706.03762
        Args:
            hp: Hyper Parameters
            encoder_vocab: vocabulary size of encoder language
            decoder_vocab: vacabulary size of decoder language
        '''
        super(Transformer, self).__init__()
        self.params = params

        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab

        # encoder
        self.encoder_embedding = Embedding(self.encoder_vocab, self.params.hidden_units, scale=True)

        if self.params.sinusoid:
            self.encoder_positional_encoding = PositionalEncoding(emb_dim=self.params.hidden_units, zeros_pad=False, scale=False)
        else:
            self.encoder_positional_encoding = Embedding(self.params.maxlen, self.params.hidden_units, zeros_pad=False, scale=False)
        self.encoder_dropout = nn.Dropout(self.params.dropout_rate)
        for i in range(self.params.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, MultiheadAttention(hidden_dim=self.params.hidden_units,
                                                                             num_heads=self.params.num_heads,
                                                                             dropout_rate=self.params.dropout_rate,
                                                                             causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, FeedForward(self.params.hidden_units,
                                                                    [4 * self.params.hidden_units,
                                                                     self.params.hidden_units]))

        # decoder
        self.decoder_embedding = Embedding(self.decoder_vocab, self.params.hidden_units, scale=True)
        if self.params.sinusoid:
            self.decoder_positional_encoding = PositionalEncoding(emb_dim=self.params.hidden_units, zeros_pad=False, scale=False)
        else:
            self.decoder_positional_encoding = Embedding(self.params.maxlen, self.params.hidden_units, zeros_pad=False, scale=False)

        self.decoder_dropout = nn.Dropout(self.params.dropout_rate)
        for i in range(self.params.num_blocks):
            self.__setattr__('dec_self_attention_%d' % i,
                             MultiheadAttention(hidden_dim=self.params.hidden_units,
                                                num_heads=self.params.num_heads,
                                                dropout_rate=self.params.dropout_rate,
                                                causality=True))
            self.__setattr__('dec_vanilla_attention_%d' % i,
                             MultiheadAttention(hidden_dim=self.params.hidden_units,
                                                num_heads=self.params.num_heads,
                                                dropout_rate=self.params.dropout_rate,
                                                causality=False))
            self.__setattr__('dec_feed_forward_%d' % i, FeedForward(self.params.hidden_units,
                                                                    [4 * self.params.hidden_units,
                                                                     self.params.hidden_units]))
        self.logits_layer = nn.Linear(self.params.hidden_units, self.decoder_vocab)
        self.label_smoothing = LabelSmoothing()
        # self.losslayer = nn.CrossEntropyLoss(reduce=False)

    def forward(self, source_batch, target_batch):
        # define decoder inputs
        if self.params.use_gpu:
            self.decoder_inputs = torch.cat([Variable(torch.ones(target_batch[:, :1].size()).cuda() * 2).long(), target_batch[:, :-1]], dim=-1)  # 2:<s>
        else:
            self.decoder_inputs = torch.cat([Variable(torch.ones(target_batch[:, :1].size()) * 2).long(), target_batch[:, :-1]], dim=-1)  # 2:<s>

        # Encoder
        self.encoder_embeded = self.encoder_embedding(source_batch)
        # Positional Encoding
        if self.params.sinusoid:
            self.encoder_embeded += self.encoder_positional_encoding(source_batch)
        else:
            if self.params.use_gpu:
                self.encoder_embeded += self.encoder_positional_encoding(
                    Variable(torch.unsqueeze(torch.arange(0, source_batch.size()[1]), 0).repeat(source_batch.size(0), 1).long().cuda()))
            else:
                self.encoder_embeded += self.encoder_positional_encoding(
                    Variable(torch.unsqueeze(torch.arange(0, source_batch.size()[1]), 0).repeat(source_batch.size(0), 1).long()))
        self.encoder_embeded = self.encoder_dropout(self.encoder_embeded)
        # Blocks
        for i in range(self.params.num_blocks):
            self.encoder_embeded = self.__getattr__('enc_self_attention_%d' % i)(self.encoder_embeded, self.encoder_embeded, self.encoder_embeded)
            # Feed Forward
            self.encoder_embeded = self.__getattr__('enc_feed_forward_%d' % i)(self.encoder_embeded)
        # Decoder
        self.decoder_embedded = self.decoder_embedding(self.decoder_inputs)
        # Positional Encoding
        if self.params.sinusoid:
            self.decoder_embedded += self.decoder_positional_encoding(self.decoder_inputs)
        else:
            if self.params.use_gpu:
                self.decoder_embedded += self.decoder_positional_encoding(
                    Variable(torch.unsqueeze(torch.arange(0, self.decoder_inputs.size()[1]), 0).repeat(self.decoder_inputs.size(0), 1).long().cuda()))
            else:
                self.decoder_embedded += self.decoder_positional_encoding(
                    Variable(torch.unsqueeze(torch.arange(0, self.decoder_inputs.size()[1]), 0).repeat(self.decoder_inputs.size(0), 1).long()))

        # Dropout
        self.decoder_embedded = self.decoder_dropout(self.decoder_embedded)
        # Blocks
        for i in range(self.params.num_blocks):
            # self-attention
            self.decoder_embedded = self.__getattr__('dec_self_attention_%d' % i)(self.decoder_embedded, self.decoder_embedded, self.decoder_embedded)
            # vanilla attention
            self.decoder_embedded = self.__getattr__('dec_vanilla_attention_%d' % i)(self.decoder_embedded, self.encoder_embeded, self.encoder_embeded)
            # feed forward
            self.decoder_embedded = self.__getattr__('dec_feed_forward_%d' % i)(self.decoder_embedded)

        # Final linear projection
        self.logits = self.logits_layer(self.decoder_embedded)
        self.probs = F.softmax(self.logits, dim=-1).view(-1, self.decoder_vocab)
        _, self.preds = torch.max(self.logits, -1)
        self.istarget = (1. - target_batch.eq(0.).float()).view(-1)
        self.acc = torch.sum(self.preds.eq(target_batch).float().view(-1) * self.istarget) / torch.sum(self.istarget)

        # Loss
        if self.params.use_gpu:
            self.y_onehot = torch.zeros(self.logits.size()[0] * self.logits.size()[1], self.decoder_vocab).cuda()
        else:
            self.y_onehot = torch.zeros(self.logits.size()[0] * self.logits.size()[1], self.decoder_vocab)

        self.y_onehot = Variable(self.y_onehot.scatter_(1, target_batch.view(-1, 1).data, 1))

        self.y_smoothed = self.label_smoothing(self.y_onehot)

        # self.loss = self.losslayer(self.probs, self.y_smoothed)
        self.loss = - torch.sum(self.y_smoothed * torch.log(self.probs), dim=-1)
        # print(self.loss)

        self.mean_loss = torch.sum(self.loss * self.istarget) / torch.sum(self.istarget)

        return self.mean_loss, self.preds, self.acc

