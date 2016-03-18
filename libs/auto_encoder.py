# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L


"""
Auto Encoder
"""


class AutoEncoder(chainer.Chain) :

    def __init__(self, in_layer, out_layer, activation, train = True) :
        super(AutoEncoder, self).__init__(
            l_in  = in_layer,
            l_out = out_layer,
        )
        self.activation = activation
        self.train      = train


    def forward_middle(self, x, activate = True) :
        if activate :
            return self.activation(self.l_in(x))
        else :
            return self.l_in(x)


    def clear(self) :
        self.loss = None

    
    def __call__(self, x) :
        self.clear()
        h = F.dropout(self.activation(self.l_in(x)), train = self.train)
        y = self.activation(self.l_out(h))

        self.loss = F.mean_squared_error(y, x)
        return self.loss


