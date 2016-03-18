# -*- coding: utf-8 -*-

import csv
import logging
import sys

import numpy as np
import six

from docopt import docopt

# Chainer MNIST data loader
# https://github.com/pfnet/chainer/blob/master/examples/mnist/data.py
from chainer_mnist import data as mnist_data

import settings as s

from libs.stacked_auto_encoder import StackedAutoEncoder


__doc__ = """
Descriptions :
  MNIST (手書き文字認識) の次元圧縮を題材とした
  Stacked Auto-Encoder の実行サンプル

Usage: {0} (-h | --help)
       {0} [-g GPU] <output>

Options :
  -h, --help   View this document.
  -g GPU       GPU ID [default: -1]
  <output>     Output coordinates CSV file.
""".format(sys.argv[0])


##################################################
### Main                                       ###
##################################################
def execute(gpu_id = -1, output_csv = "output.csv") :

    ### Settings
    log_level   = s.LOG_LEVEL

    layer_sizes            = s.LAYER_SIZES
    activation_type        = s.ACTIVATION_TYPE
    optimizer_type         = s.OPTIMIZER_TYPE
    pretraining_batch_size = s.PRETRAINING_BATCH_SIZE
    pretraining_epochs     = s.PRETRAINING_EPOCHS
    finetuning_batch_size  = s.FINETUNING_BATCH_SIZE
    finetuning_epochs      = s.FINETUNING_EPOCHS
    forward_batch_size     = s.FORWARD_BATCH_SIZE

    ### Logger settings
    logger    = logging.getLogger("")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    ### Load MNIST data
    mnist = mnist_data.load_mnist_data()
    x_all = mnist['data'].astype(np.float32) / 255 # 0 ～ 1
    y_all = mnist['target'].astype(np.int32)


    ### Set Train and Test data
    train_size = 60000
    test_size  = 10000
    x_train, x_test = np.split(x_all, [train_size])
    y_train, y_test = np.split(y_all, [train_size])


    ### Layer settings
    SAE = StackedAutoEncoder(layer_sizes, gpu_id, logger,
                             {'activation_type': activation_type,
                              'optimizer_type': optimizer_type})


    ### Pretraining
    SAE.pretraining(x_train, pretraining_batch_size, pretraining_epochs)

    
    ### Finetuning
    SAE.finetuning(x_train, finetuning_batch_size, finetuning_epochs)


    ### Write results
    coords = SAE.forward_all(x_test, forward_batch_size)
    labels = y_test

    f      = open(output_csv, 'w')
    writer = csv.writer(f, lineterminator = "\n")
    for i in six.moves.range(len(labels)) :
        writer.writerow([i, labels[i], coords[i][0], coords[i][1]])
    f.close()

    print("Complete!!!")


if __name__ == '__main__' :

    args = docopt(__doc__)
    gpu_id       = int(args['-g'])
    output_csv   = args['<output>']

    execute(gpu_id, output_csv)
