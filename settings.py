# -*- coding: utf-8 -*-

import logging

"""
アプリケーション設定
"""


##################################################
### General                                    ###
##################################################

### Log level
# LOG_LEVEL = logging.DEBUG
LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.ERROR


##################################################
### General                                    ###
##################################################

### Layer settings
INPUT_SIZE  = 28 * 28 # = 784
OUTPUT_SIZE = 2
LAYER_SIZES = [INPUT_SIZE, 1000, 500, 250, OUTPUT_SIZE]

### Activation
ACTIVATION_TYPE = 'Sigmoid'
# ACTIVATION_TYPE = 'ReLU'

### Optimizer
# OPTIMIZER_TYPE = 'SGD'
OPTIMIZER_TYPE = 'Adam'

### Pretraining
PRETRAINING_BATCH_SIZE = 100
PRETRAINING_EPOCHS     = 30

### Finetuning
FINETUNING_BATCH_SIZE = 100
FINETUNING_EPOCHS     = 30

### Forward all
FORWARD_BATCH_SIZE = 100
