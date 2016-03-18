# -*- coding: utf-8 -*-

import math

import numpy as np
import six

import chainer
import chainer.functions as F
import chainer.links as L

from chainer import cuda
from chainer import optimizers
from chainer import serializers


from .auto_encoder import AutoEncoder


class StackedAutoEncoder(object) :
    """
    Stacked Auto Encoder (実行部)
    """

    def __init__(self, layer_sizes, gpu_id = -1, logger = None, params = {}) :
        self.layer_sizes = layer_sizes
        self.gpu_id      = gpu_id
        self.logger      = logger

        # Activation
        activation_type = params.get('activation_type', 'Sigmoid')
        self.activation = F.sigmoid
        if activation_type == 'ReLU' :
            self.activation = F.relu

        # Optimizer
        optimizer_type = params.get('optimizer_type', 'SGD')
        self.optimizer = optimizers.MomentumSGD()
        if optimizer_type == 'Adam' :
            self.optimizer = optimizers.Adam()

        # Layers
        self.f_layers = self.__createLayers(layer_sizes)
        self.b_layers = self.__createLayers(layer_sizes, True)


    def pretraining(self, dataset, batch_size, epochs) :
        xp = cuda.cupy if self.gpu_id >= 0 else np

        if self.logger :
            self.logger.info("Start pretraining...")

        ### Copy layers
        # NOTE: https://github.com/pfnet/chainer/issues/715
        self.f_layers = self.__copyLayers(self.f_layers)
        self.b_layers = self.__copyLayers(self.b_layers)

        pre_train_data = None
        for l_idx in six.moves.range(len(self.f_layers)) :
            if self.logger :
                self.logger.info("Pretraining {0} and {1} layers.".format(l_idx + 1, l_idx + 2))

            ### データ設定
            train_data     = None
            if l_idx == 0 :
                train_data = dataset
            else :
                train_data = pre_train_data

            ### Chain 作成
            model = AutoEncoder(self.f_layers[l_idx], self.b_layers[l_idx], self.activation)
            if self.gpu_id >= 0 :
                model.to_gpu()
            self.optimizer.setup(model)

            ### Training
            train_size = len(train_data)
            batch_max  = int(math.ceil(train_size / float(batch_size)))

            epoch = 0
            while True :
                epoch    += 1
                indexes   = np.random.permutation(train_size)
                sum_loss  = 0

                ### Batch training
                for i in six.moves.range(batch_max) :
                    start   = (i + 0) * batch_size
                    end     = (i + 1) * batch_size
                    x_batch = chainer.Variable(xp.array(train_data[indexes[start:end]], dtype = xp.float32))

                    self.optimizer.update(model, x_batch)
                    sum_loss += model.loss.data

                epoch_loss = sum_loss / batch_max
                if self.logger :
                    self.logger.info("Epoch {0}, Training loss = {1}".format(epoch, epoch_loss))

                if epoch >= epochs :
                    break

            ### Set next layer data
            pre_train_data = np.zeros((train_size, self.layer_sizes[l_idx + 1]), dtype = train_data.dtype)
            c_model = model.copy()
            c_model.train = False

            for i in six.moves.range(batch_max) :
                start   = (i + 0) * batch_size
                end     = (i + 1) * batch_size
                x_batch = chainer.Variable(xp.array(train_data[start:end], dtype = xp.float32))
                y_batch = c_model.forward_middle(x_batch, activate = True).data
                if self.gpu_id >= 0 :
                    y_batch = cuda.to_cpu(y_batch) # NOTE: ランダムデータ投入の permutation は CPU Only なので、to_cpu している。

                pre_train_data[start:end] = y_batch

        if self.logger :
            self.logger.info("Complete pretraining.")

                    
    def finetuning(self, dataset, batch_size, epochs) :
        xp = cuda.cupy if self.gpu_id >= 0 else np

        if self.logger :
            self.logger.info("Start finetuning...")

        ### Copy layers
        # NOTE: https://github.com/pfnet/chainer/issues/715
        self.f_layers = self.__copyLayers(self.f_layers)
        self.b_layers = self.__copyLayers(self.b_layers)

        ### Chain 作成
        model = StackedAutoEncoderFinetuning(self.f_layers, self.b_layers, self.activation)
        if self.gpu_id >= 0 :
            model.to_gpu()
        self.optimizer.setup(model)

        ### Training
        train_data = dataset
        train_size = len(train_data)
        batch_max  = int(math.ceil(train_size / float(batch_size)))

        epoch = 0
        while True :
            epoch    += 1
            indexes   = np.random.permutation(train_size)
            sum_loss  = 0

            ### Batch training
            for i in six.moves.range(batch_max) :
                start   = (i + 0) * batch_size
                end     = (i + 1) * batch_size
                x_batch = chainer.Variable(xp.array(train_data[indexes[start:end]], dtype = xp.float32))

                self.optimizer.update(model, x_batch)
                sum_loss += model.loss.data

            epoch_loss = sum_loss / batch_max
            if self.logger :
                self.logger.info("Epoch {0}, Training loss = {1}".format(epoch, epoch_loss))

            if epoch >= epochs :
                break

        if self.logger :
            self.logger.info("Complete finetuning.")


    def forward_all(self, dataset, batch_size) :
        xp = cuda.cupy if self.gpu_id >= 0 else np

        if self.logger :
            self.logger.info("Get result start.")

        ### Copy layers
        # NOTE: https://github.com/pfnet/chainer/issues/715
        self.f_layers = self.__copyLayers(self.f_layers)
        self.b_layers = self.__copyLayers(self.b_layers)

        ### Chain 作成
        model = StackedAutoEncoderForwardAll(self.f_layers, self.activation)
        if self.gpu_id >= 0 :
            model.to_gpu()

        ### 結果取得
        test_data = dataset
        test_size = len(test_data)
        batch_max = int(math.ceil(test_size / float(batch_size)))

        y_data = np.zeros((test_size, self.layer_sizes[len(self.layer_sizes) - 1]), dtype = test_data.dtype)
        for i in six.moves.range(batch_max) :
            start   =  i      * batch_size
            end     = (i + 1) * batch_size
            x_batch = chainer.Variable(xp.array(test_data[start:end], dtype = xp.float32))
            y_batch = model(x_batch).data
            if self.gpu_id >= 0 :
                y_batch = cuda.to_cpu(y_batch)
            y_data[start:end] = y_batch
            
        if self.logger :
            self.logger.info("Complete get result.")

        return y_data

    
    def __createLayers(self, layer_sizes, backward = False) :
        """
        (private) 連続したサイズ指定により、複数の Linear layer 群を作成。
        """
        layers = []
        for i in six.moves.range(len(layer_sizes) - 1) :
            if not backward :
                in_size  = layer_sizes[i]
                out_size = layer_sizes[i + 1]
            else :
                in_size  = layer_sizes[i + 1]
                out_size = layer_sizes[i]
            layers.append(F.Linear(in_size, out_size))

        return layers


    def __copyLayers(self, org_layers) :
        """
        (private) Linear layer 群のコピー
        # NOTE: 現状では 1つの Link は 1つの Chain にしかつなぐことができないための対応。
        #       https://github.com/pfnet/chainer/issues/715
        """
        new_layers = []
        for layer in org_layers :
            new_layers.append(layer.copy())

        return new_layers


class StackedAutoEncoderFinetuning(chainer.Chain) :
    """
    Stacked Auto Encoder (Finetuning Chain)
    """

    def __init__(self, f_layers, b_layers, activation, train = True) :
        super(StackedAutoEncoderFinetuning, self).__init__()
        self.f_layers   = f_layers
        self.b_layers   = b_layers
        self.activation = activation
        self.train    = train
        
        for num, f_layer in enumerate(f_layers, 1) :
            self.add_link("l_f{0}".format(num), f_layer)
        for num, b_layer in enumerate(b_layers, 1) :
            self.add_link("l_b{0}".format(num), b_layer)

        self.len_f_layers = len(self.f_layers)
        self.len_b_layers = len(self.b_layers)
            

    def clear(self) :
        self.loss = None
            

    def __call__(self, x) :
        self.clear()

        h = x
        for num in six.moves.range(1, self.len_f_layers + 1) :
            h = F.dropout(self.activation(self.__getitem__("l_f{0}".format(num))(h)), train = self.train)
        for num in reversed(six.moves.range(2, self.len_b_layers + 1)) :
            h = F.dropout(self.activation(self.__getitem__("l_b{0}".format(num))(h)), train = self.train)
        y = self.__getitem__("l_b{0}".format(1))(h)

        self.loss = F.mean_squared_error(y, x)
        return self.loss


class StackedAutoEncoderForwardAll(chainer.Chain) :
    """
    Stacked Auto Encoder (Forward all Chain)
    """

    def __init__(self, f_layers, activation, train = False) :
        super(StackedAutoEncoderForwardAll, self).__init__()
        self.f_layers   = f_layers
        self.activation = activation
        self.train    = train
        
        for num, f_layer in enumerate(f_layers, 1) :
            self.add_link("l_f{0}".format(num), f_layer)
 
        self.len_f_layers = len(self.f_layers)
             

    def clear(self) :
        self.loss = None
            

    def __call__(self, x) :
        self.clear()

        h = x
        for num in six.moves.range(1, self.len_f_layers) :

            h = F.dropout(self.activation(self.__getitem__("l_f{0}".format(num))(h)), train = self.train)
        y = self.__getitem__("l_f{0}".format(self.len_f_layers))(h)
        return y



